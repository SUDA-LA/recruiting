# New-stu-training
## 1 李正华TODO
 我的PPT不好，有的要简化，例如GB编码没有具体说明：两个字节，第一个字节怎么样。  
 有的需要更详细、更丰满。
## 2 说明
* 所有俱乐部同学，包括刚进入实验室的研究生，都要做NLP基础编程练习，一方面提高编程能力，另一方面打好机器学习和自然语言处理的基础
* 我们精心设计了一系列的基础练习作业，从简单到复杂，逐步深入：[题目（课程）主页](#training)
* 请直入主题，从题目出发，按照推荐的顺序，逐一做
* 遇到一个题目，去下面的课件或讲义中寻找相关内容，进行快速学习和理解
  * 做基础练习的时候，要关注主要知识点，不要陷入细枝末节，否则的话，进度就太慢了。比如随机过程，这个本身是非常大的一个概念，千万不要钻进去学习，了解其基本概念、例子即可。
  * 总之，一切以正确完成编程题目为主要目标，先做出来，然后做对，最后做好
  * 我相信，等编程作业做完了，很多知识点就会有自己的理解了
* 注意，**千万不要直接看别人写好的代码，一定要努力自己去理解和消化。** 一定要有一个自己思考的过程，尽最大努力自己写代码，即使效率低，准确率差，也没关系。通过自己的思考，逐渐优化和提高。这个过程非常重要，很有意义。
* 另外，**尽量不要在网上找各种参考资料。** 虽然我的讲义很精简，但是如果仔细看，认真思考，推导，一定能搞明白。这个过程也很重要，对自我提升很有帮助。
  * 任何一个理论或方法，从不同角度都可以进行解释，看的角度多了，就没有自己的角度了。网上的各种资料，都是从不同角度来说同一个事，看多了就乱了。
* 如果有问题，可以找自己的mentor去讨论，如果mentor也不懂，那说明mentor也没理解，就可以发邮件给老师。
## 3 参考书目
* Dan Jurafsky. [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/),[中文翻译](https://www.kancloud.cn/drxgz/slp20201230#/dashboard)(强烈推荐！)
* [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/)（李正华强烈推荐，看完前三章就差不多了）
* Andrew Ng的视频:[吴恩达深度学习(带中文字幕)](https://mooc.study.163.com/university/deeplearning_ai#/c)
* Chris Manning. 2005. 统计自然语言处理基础.
* 宗成庆. 2008. 统计自然语言处理.
* 李航. 2012. 统计学习方法.
* 神经网络与深度学习
* ...  

**神经网络公开课第十三章：面向自然语言处理的神经网络 （苏州大学 李正华）**
* [第1节：从离散特征到连续稠密向量表示](http://hlt.suda.edu.cn/~zhli/NLP-DL/13.1.mp4)
* [第2节：表示学习](http://hlt.suda.edu.cn/~zhli/NLP-DL/13.2.mp4)
* [第3节：序列标注问题](http://hlt.suda.edu.cn/~zhli/NLP-DL/13.3.mp4)
* [第4节：句法树解析问题](http://hlt.suda.edu.cn/~zhli/NLP-DL/13.4.mp4)
## 4 数据（utf8编码）
### 4.1 CoNLL格式的含义
* 每个词占一行，每行的第2列为当前词语，第4列为当前词的词性，第7列为当前词的中心词的序号，第8列为当前词语与中心词的依存关系。句子与句子之间以空行间隔。
### 4.2 最大匹配分词数据
* [分词数据](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E6%9C%80%E5%A4%A7%E5%8C%B9%E9%85%8D%E5%88%86%E8%AF%8D%E6%95%B0%E6%8D%AE/data.conll)
### 4.3 词性标注数据
* [小数据集](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E8%AF%8D%E6%80%A7%E6%A0%87%E6%B3%A8%E6%95%B0%E6%8D%AE/data.tar.gz)  

 | 训练 | 开发 |     
 | :----: | :----: |   
 | 803句 | 1910句 |  
* [大数据集](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E8%AF%8D%E6%80%A7%E6%A0%87%E6%B3%A8%E6%95%B0%E6%8D%AE/ctb5-postagged.tar.gz)  

| 训练 | 开发 | 测试 |    
| :----: | :----: | :----: |  
| 16091句 | 803句 | 1910句 |  
* 示例： 输入：严守一 把 手机 关 了 输出：严守一/NR 把/P手机/NN关/VV 了/SP
<a id="table1">Table - 1</a>
## 5 <a id="training">基础编程训练列表</a>
### 5.1 分字
* 2022春IR课程视频和图片：
  * 低画质：[作业1](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/split-char-low-quality.mp4)
  * 高画质：[作业1-part1](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/split-char-part-1.mp4)、[作业1-part2](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/split-char-part-2.mp4)
  * 图片：[图1](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/split-char-figure-1.jpg)、[图2](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/split-char-figure-2.jpg)
* UTF-8数据：[文件:Sentence.txt]([http://hlt.suda.edu.cn/index.php/%E6%96%87%E4%BB%B6:Sentence.txt](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E6%9C%80%E5%A4%A7%E5%8C%B9%E9%85%8D%E5%88%86%E8%AF%8D%E6%95%B0%E6%8D%AE/Sentence.txt))
* UFT-8编码规则：
```
1字节 0xxxxxxx
2字节 110xxxxx 10xxxxxx
3字节 1110xxxx 10xxxxxx 10xxxxxx
4字节 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
5字节 111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
6字节 1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 
```
* 下面的内容可以不看

  * 给定文件，将文件中的句子按照字（字符）切分，字符中间用空格隔开。用C/C++实现。Python（3.0）可以直接用split处理UTF8编码的字符串，也试试，对比一下结果。  
  * 参考资料：[文件:Chinese-encoding.pdf](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E8%AE%B2%E4%B9%89/Chinese-encoding.pdf)
  * 数据：[几个不同编码的文件](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E5%88%86%E5%AD%97%E6%95%B0%E6%8D%AE/example.tar.gz)，可以用hexdump查看。也可以自己生成不同编码的文件。
### 5.2 最大匹配分词
* 2022春IR课程视频和图片：
  * 低画质：[作业3](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/word-seg-max-match-low-quality.mp4)
  * 高画质：[作业3](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/word-seg-max-match.mp4)
  * 图片：[图](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/word-seg-max-match.jpg)
* 数据下载：
  * 字典：[文件:Dict.txt](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E6%9C%80%E5%A4%A7%E5%8C%B9%E9%85%8D%E5%88%86%E8%AF%8D%E6%95%B0%E6%8D%AE/Dict.txt)
  * 待分词：[文件:Sentence.txt](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E6%9C%80%E5%A4%A7%E5%8C%B9%E9%85%8D%E5%88%86%E8%AF%8D%E6%95%B0%E6%8D%AE/Sentence.txt)
  * 正确答案（人工标注的，你的模型的预测结果要和这个文件进行对比，从而得到P/R/F值）：[文件:Answer.txt](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E6%9C%80%E5%A4%A7%E5%8C%B9%E9%85%8D%E5%88%86%E8%AF%8D%E6%95%B0%E6%8D%AE/Answer.txt);
  * 正向最大匹配分词模型的预测结果（如果你的程序写对了，那么应该和这个结果一模一样）：[文件:Out.txt](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E6%9C%80%E5%A4%A7%E5%8C%B9%E9%85%8D%E5%88%86%E8%AF%8D%E6%95%B0%E6%8D%AE/Out.txt)
```
*正确实验结果   
**正确识别的词数：20263   
**识别出的总体个数：20397   
**测试集中的总体个数：20454   
**正确率：0.99343  
**召回率：0.99066   
**F值：0.99204  
```
* 下面的内容可以不看
* 参考课件：[最大匹配](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E8%AE%B2%E4%B9%89/max-match.ppt)
### 5.3 有监督HMM词性标注
* 图片和视频：
  * 低画质：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-1.mp4)、[第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-2.mp4)、[第3部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-3.mp4)、[第4部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-4.mp4)
  * 高画质：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-1-hd.mp4)、[第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-2-hd.mp4)、[第3部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-3-hd.mp4)、[第4部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-4-hd.mp4)
  * 图片：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-1.jpg)、[第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-2.jpg)、[第3部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-3.jpg)、[第4部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-4.jpg)
* 参考课件：[Collins教授课件](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/collins-tagging.pdf)、[李正华的课件](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E8%AE%B2%E4%B9%89/HMM.pdf)、[理解HMM的Viterbi](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E8%AE%B2%E4%B9%89/HMM-v2.pptx)、[HMM模型中极大似然估计的由来(公式推导)](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E8%AE%B2%E4%B9%89/HMM%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1.pdf)
### 5.4 基于线性模型（linear model）的词性标注
* 要点：判别模型、partial feature
* 参考课件：[李正华老师课件](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E8%AE%B2%E4%B9%89/LinearModel.pdf)
* 图片和视频：
  * 视频：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-1.mp4)、[第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-2.mp4)、[第3部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-3.mp4)、[第4部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-4.mp4)、[第5部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-5.mp4)
  * 图片：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-1.jpg)、[第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-2.jpg)、[第3部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-3.jpg)、[第4部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-4.jpg)、[第5部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/linear-model-5.jpg)
### 5.5 基于最大熵（max-entropy，log-linear）模型的词性标注
* 要点：梯度下降方法，Adam优化
* 参考课件：[李正华老师课件](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E8%AE%B2%E4%B9%89/LogLinearModel.pdf)、[Collins教授课件](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99/collins-loglinear.pdf)
* 图片和视频：
  * 视频：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/Maximum-entropy-1.mp4)、[第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/Maximum-entropy-2.mp4)
  * 图片：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/Maximum-entropy-1.jpg)、[第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/Maximum-entropy-2.jpg)
### 5.6 基于全局线性模型（global linear model）的词性标注
参考课件：[李正华老师课件](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E8%AE%B2%E4%B9%89/GlobalLinearModel.pdf)
### 5.7 基于条件随机场（conditional random field，CRF）模型的词性标注
* 要点：全局概率、期望、Forward-backward结合、viterbi解码
* 参考课件：[李正华老师课件](https://github.com/SUDA-LA/recruiting/blob/main/New-stu-training/%E8%AE%B2%E4%B9%89/CRF.pdf)
### 5.8 基于前馈神经网络（FFN）的词性标注
* 要点：必须自己实现前向计算loss，和backpropagation。
* 参考：neural networks and deeplearning神经网络入门书籍([英文版](http://neuralnetworksanddeeplearning.com/)、[中文版](https://github.com/zhanggyb/nndl/releases/download/latest/nndl-ebook.pdf))基本阅读完前三章即可完成本任务. [吴恩达深度学习(带中文字幕)](https://mooc.study.163.com/university/deeplearning_ai#/c)
### 5.9 基于FFN-CRF的词性标注
* 要点：仍然自己实现前向计算loss，和backpropagation。
* 提示：将神经网络输出看成发射矩阵，之后加上转移矩阵
### 5.10 基于BiLSTM的词性标注
* 要点：可以利用Pytorch自带的。Dropout等的使用，是关键。
### 5.11 基于BiLSTM-CRF的词性标注
### 5.12 github已有代码，不同同学的代码可以看不同的branch：
[github网址](https://github.com/SUDA-LA/CIP)
## 6 后续自主学习扩展
### 6.1 基于图的依存句法分析
直接用神经网络实现即可。Biaffine Parser框架。  
重点：Eisner动态规划解码算法（看我的COLING-2014 tutorial）  
进而可以扩展到TreeCRF，将Eisner算法扩展为inside算法。  
具体看我们的ACL-2020文章：Yu Zhang et al.
### 6.2 基于转移的依存句法分析
了解一下转移系统  
### 6.3 Seq2Seq (RNN) NMT with attention
了解一下语言生成  
### 6.4 Transformer NMT
这里面的技术细节很多。  
### 6.5 无监督学习
HMM-EM  
VAE  
### 6.6 ELMo/BERT的原理



