from setuptools import setup, find_packages

setup(
    name="tgb_seq",  # 你包的名字
    version="0.1",   # 版本号
    packages=find_packages(),  # 自动找到子包
    install_requires=[
        "numpy",
        "torch",
        # 其他依赖
    ],
    author="Your Name",
    description="Temporal Graph Benchmark - Sequential Dynamics",
)
