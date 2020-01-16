#data->[instance]
from typing import Dict, Iterable, Iterator, Union, List

import torch
import torch.nn as nn
import torch.nn.modules
import torch.nn.functional as f
from allennlp.nn import util
from allennlp.predictors import Predictor
from torch import optim

from allennlp.data.dataset import Batch
from allennlp.data.iterators.data_iterator import TensorDict
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder,BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder,PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training import Trainer
from overrides import overrides
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data import Instance, DataIterator
from allennlp.data.tokenizers import Token,WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.iterators import BucketIterator,BasicIterator
from allennlp.data.vocabulary import Vocabulary
# 自己写一个toker，将字符串序列转化为tokens
from tqdm import tqdm

use_gpu=False
if torch.cuda.is_available():
     use_gpu=True
emb_length=256
voc_length=20000
generating_length=10
train_batch=8
eval_batch=10
class MyWordToker():
    def __init__(self):
        pass
    def tokenize(self,text:str)->[Token]:
        return[ Token(x) for x in text.split()]
DatasetReader.register('textreader')
class DataReader(DatasetReader):
    def __init__(self,toker,tokindexer,targetindexer,lazy=False):
        super().__init__(lazy=lazy)
        self.toker=toker
        self.tokindexer=tokindexer
        self.targetindexer=targetindexer
    @overrides
    def text_to_instance(self,tokens,target)->Instance:
        field={"tokens":TextField(tokens,self.tokindexer)}
        field["target"]=TextField(target,self.targetindexer)
        return Instance(field)
    @overrides
    def _read(self,path:str,debug=True)->Iterable[Instance]:
        #pytorch这里是不分句的，可以认为整个txt文件就是一个示例，不过分batch是可以的。
        #分句也可以，分句的话就认为一句话是一个instance，或者干脆认为一行是一个instance，不加eos，而是学习标点符号。
        eva=False
        if not (path.find("valid") ==-1):
           eva=True
        lines=0
        with open(path,'r') as f:
            for line in f:

                line+="<EOS>"
                tokens=self.toker.tokenize(text=line)
                if len(tokens) == 1:
                    continue
                lines += 1
                if eva and lines == 10:
                     break
                if debug and lines==1000:
                    break
                #这里至少instance长度为2，否则会导致空
                source=tokens[:-1]
                target=tokens[1:]
                yield self.text_to_instance(source,target)



@DataIterator.register('wholeset_iterator')
class WholeSetIterator(DataIterator):
    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True) -> Iterator[TensorDict]:
        Batch=self._create_batches(instances,shuffle)
        for batch in Batch:
            batch.index_instances(self.vocab)
            yield batch.as_tensor_dict()
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        yield Batch(instances)


@Model.register("KILM")
#所有的NLP模型都会先做embedding，将输入的instance映射到embedding向量
class KILM(Model):
    def __init__(self,
                 vocab:Vocabulary=None,
                 emb:TextFieldEmbedder=None,
                 rnn:Seq2SeqEncoder=None,
                 decoder=None,
                 loss_function=None,
                 generator=None):
        super(KILM,self).__init__(vocab)
        self.emb=emb
        self.rnn=rnn
        self.decoder=decoder
        self.loss_function=loss_function
        self.generator=generator or nn.GRU(emb_length,emb_length,batch_first=True)
    #一个forward是对一个batch而言，即训练的最小单位
    #同一个batch默认padding为相同长度
    def forward(self,
                tokens:Dict[str,torch.Tensor],
                target:Dict[str,torch.Tensor]=None,#如果训练 传入target
                length:int=0 #如果预测 传入length
                )->Union[List[str],Dict[str,torch.Tensor]]:

        mask=get_text_field_mask(tokens)
        emb_value=self.emb(tokens)
        # shape[batch,seq,emb]
        out=self.rnn(emb_value,mask)
        if length >0:
            batch_word=[]
            #[batch,1,emb]
            prev=out[:,-2:-1]
            #[1,batch,emb]
            #hidden必须要求batch在二维，做了permute转二维又必须contiguous，有没有其他办法？
            hidden=prev.permute(1,0,2).contiguous()
            for i in range(length):

                #[batch,emb]
                word=f.softmax(prev,dim=2)
                #[batch,1]
                word_inx=word.max(dim=2)[1]
                batch_word.append([self.vocab.get_token_from_index(index.item())for index in word_inx])

                #TODO Attention Decoder
                prev,hidden=self.generator(prev,hidden)
            return batch_word
        logit=self.decoder(out)
        #target_value=self.emb(target)
        target=target['tokens']
        loss=self.loss_function(logit,target,mask)

        return {"logit":out,"loss":loss}
    #我们可以揣摩一下为什么allennlp如此器重字典：作为一种高层抽象

Predictor.register("lm_predictor")
class LMPredictor(Predictor):
    def __init__(self,
                 model:Model,
                 Source:Iterable[Instance],
                 it:DataIterator,
                 device=-1):
        self.model=model
        self.it=it
        self.source=Source
        self.device=device
    # 从model中拿出一个instance，预测之后的length个词

    def predict(self,
                length,
                outputpath:str):
        #用train的风格写
        #首先拿到一个batch,这里必须手动设置num epoch，否则不会停止
        batches=self.it(self.source,num_epochs=1)
        batches=tqdm(batches,total=self.it.get_num_batches(self.source))
        with torch.no_grad() and open(outputpath,'w') as f:
            for batch in batches:
                #在batch创建出来的时候需要move
                batch=util.move_to_device(batch,self.device)
                #batch长度的列表，列表：[generating_legnth]个str
                words=self.model.forward(**batch,length=length)
                f.writelines(''.join(sequence)+'\n' for sequence in words)

if __name__=="__main__":
    wt=MyWordToker()
    #这两个indexer并行工作，最终得到两个tensor
    ti={"tokens":SingleIdTokenIndexer()}
    reader=DataReader(wt,ti,ti)
    A,B=(reader.read(file) for file in ["../data/wikitext-2/train.txt","../data/wikitext-2/valid.txt"])
    #TODO:探究参数和override
    #B=reader.myread("../data/wikitext-2/valid.txt")
   # print(vars(A[1].fields["token"]))
    vocab=Vocabulary.from_instances(A,max_vocab_size=voc_length)
   # print(vocab._index_to_token)
    it=BucketIterator(sorting_keys=[("tokens","num_tokens")],batch_size=train_batch)
    it2=BucketIterator(sorting_keys=[("tokens", "num_tokens")], batch_size=eval_batch)
    # it2=WholeSetIterator()
    #it2.index_with(vocab)
    it.index_with(vocab)
    it2.index_with(vocab)
    #  batch=next(iter(it(A)))
   # print(batch)

    token_emb = Embedding(num_embeddings=voc_length, embedding_dim=emb_length, padding_index=0)
    word_emb = BasicTextFieldEmbedder({"tokens": token_emb})
    rnn=PytorchSeq2SeqWrapper(nn.LSTM(emb_length,emb_length,batch_first=True))
    decoder=nn.Linear(emb_length,voc_length)
    loss=sequence_cross_entropy_with_logits

    KILM = KILM(vocab,word_emb,rnn,decoder,loss)
    #model.cuda
    if use_gpu:
        KILM.cuda()

    trainer=Trainer(model=KILM,
                    optimizer=optim.Adam(KILM.parameters(),lr=0.01),
                    iterator=it,
                    train_dataset=A,
                    cuda_device=0 if use_gpu else -1,
                    num_epochs=500,
                    shuffle=False
                    )

    trainer.train()

    predictor=LMPredictor(KILM,B,it2,device=0 if use_gpu else -1)
    predictor.predict(10,'output.txt')