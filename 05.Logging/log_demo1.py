import logging

# 1. 基础配置（设置日志级别和格式）
logging.basicConfig(
    level=logging.DEBUG,  # 输出DEBUG及以上级别的日志
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)

# 2. 记录不同级别的日志
logging.debug('这是DEBUG级别的日志（调试信息）')
logging.info('这是INFO级别的日志（正常运行）')
logging.warning('这是WARNING级别的日志（警告）')
logging.error('这是ERROR级别的日志（错误）')
logging.critical('这是CRITICAL级别的日志（严重错误）')