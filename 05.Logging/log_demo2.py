import logging
from logging import StreamHandler  # 导入控制台处理器

# 1. 创建日志器
logger = logging.getLogger("dual_log_demo")
logger.setLevel(logging.DEBUG)  # 日志器总级别：最低接受 DEBUG（需低于所有处理器级别）

# 2. 创建文件处理器（写入 app.log）
file_handler = logging.FileHandler(
    filename=r"D:\Min\Projects\VSCodeProjects\01.Useful_Tools\05.Logging\log\app.log",
    mode="a",
    encoding="utf-8"
)
file_handler.setLevel(logging.INFO)  # 文件：只记录 INFO 及以上

# 3. 创建控制台处理器（输出到终端）
console_handler = StreamHandler()  # 默认输出到 sys.stderr（终端）
console_handler.setLevel(logging.DEBUG)  # 终端：记录 DEBUG 及以上（更详细）

# 4. 定义日志格式（可给终端和文件设置不同格式，这里统一格式）
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)  # 控制台也使用相同格式

# 5. 将两个处理器都添加到日志器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 6. 测试日志输出
logger.debug("这是 DEBUG 日志（仅终端显示，文件不记录）")
logger.info("程序启动成功（终端和文件都记录）")
logger.warning("内存使用率超过 80%（终端和文件都记录）")
try:
    1 / 0
except ZeroDivisionError:
    logger.error("除法错误", exc_info=True)  # 终端和文件都记录，含堆栈信息
logger.critical("服务器连接中断！（终端和文件都记录）")