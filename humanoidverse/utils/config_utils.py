import math
from loguru import logger
from omegaconf import OmegaConf

try:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
    OmegaConf.register_new_resolver("sum", lambda x: sum(x))
    OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
    OmegaConf.register_new_resolver("int", lambda x: int(x))
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
except Exception as e:
    logger.warning(f"Warning: Some resolvers already registered: {e}")
    
# OmegaConf.register_new_resolver。该函数用于给 OmegaConf 注册“自定义表达式解析器”。
# 注册后，你可以在 yaml 配置文件里用 ${resolver_name:参数} 的语法，动态计算值。
# 例如：${sqrt:9} 会被自动替换为 3.0
# ${if:true, 1, 0} 会被替换为 1
# ${sum:[1,2,3]} 会被替换为 6
# 这样，配置文件就支持了“条件判断、数学运算、字符串比较、列表求和”等高级表达式。


# OmegaConf.register_new_resolver("eval", eval)
# OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
# OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
# OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
# OmegaConf.register_new_resolver("sum", lambda x: sum(x))
# OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
# OmegaConf.register_new_resolver("int", lambda x: int(x))
# OmegaConf.register_new_resolver("len", lambda x: len(x))
# OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
