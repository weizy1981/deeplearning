import nltk
import ssl

# 取消SSl认证
ssl._create_default_https_context = ssl._create_unverified_context

# 下载nltk数据包
nltk.download()