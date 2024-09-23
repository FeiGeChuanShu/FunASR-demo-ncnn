# FunASR-demo-ncnn
some ncnn demos of [FunASR](https://github.com/modelscope/FunASR)  

## asr model support:  
- [x] 1.[SenseVoiceSmall](https://www.modelscope.cn/models/iic/SenseVoiceSmall)  
- [x] 3.[Paraformer-small](https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1)  
- [ ] 2.[Paraformer-large](https://www.modelscope.cn/models/dengcunqin/speech_seaco_paraformer_large_asr_nat-zh-cantonese-en-16k-common-vocab11666-pytorch)  


## vad model support:  
- [x] 1.[silero-vad v4](https://github.com/snakers4/silero-vad)  
- [x] 2.[silero-vad v5](https://github.com/snakers4/silero-vad)  

## punct model support:  
- [x] 1.[ct-transformer](https://www.modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch)   

## Run  
```
mkdir -p build
cd build
git submodule update --init
cmake ..
make
./bin/sensevoice_demo ../data/audio.wav ../models/

```
## Result  
```
$ ./bin/sensevoice_demo ../data/audio.wav ../models/
1.vad init done
2.asr init done
3.punct init done
4.load wav done
5.vad segment done
speech segment count: 2
starting recognize
0 in [0:1] done
1 in [1:2] done
6.recognize done
output result:
[0.002 : 1.182]: 再苦一苦百姓。
[1.506 : 2.699]: 日子会好起来的。

```

## Reference  
1.https://github.com/Tencent/ncnn  
2.https://github.com/csukuangfj/kaldi-native-fbank  
3.https://github.com/modelscope/FunASR  
4.https://github.com/snakers4/silero-vad