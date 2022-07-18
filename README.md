# HRankFL

## 开始之前
**python3.6.15**环境下运行代码前需要提前安装实现基于的第三方库，通过pip安装的命令如下：
```shell
pip install torch torchvision torchaudio

pip install thop onnx fedlab PyHessian scikit-learn seaborn
```

## 使用说明
1. 参考share目录下的default_config.yml文件给出程序运行的配置参数，
其中有些是必要的参数，有些参数不配置会给出默认参数，请详细参考表格。
当然对于特定的一些实验，给出了推荐的配置参数文件，对应其他yml。

2. 修改custom_path.py文件下对应路径参数，考虑到模型参数和数据集占用存储较大，
这两块内容应由使用者预先准备好，并指定出对应位置。

   
## 开发上的规范

1. test_unit.py可提供单元测试或是对外接口

2. 每个包对外提供的接口只保留在test_unit.py中，所有外部包只调用该py包的接口获得服务


## 怎样高效地进行扩展，针对其他模型和其他类型的数据集

如果想要更深层地优化代码，可以访问https://wolfsion.github.io参考整体工程架构介绍，以便快速熟悉












