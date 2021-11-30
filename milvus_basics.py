import random
from ann_benchmarks.algorithms.milvus2 import Milvus
import numpy as np
# insert
data_size= 10_000
data = [[random.random() for _ in range(2)] for _ in range(data_size)]
# search
milvus = Milvus('L2', "IVF_FLAT", nlist=10, collection_name='test1')
milvus.fit(np.array(data))
milvus.set_query_arguments(1)
search_data = [[1.0, 1.0]]
top_k = 2
results = milvus.query(np.array(search_data), top_k)
print(results)
