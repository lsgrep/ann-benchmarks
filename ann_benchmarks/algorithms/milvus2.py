from __future__ import absolute_import

import os

import numpy
import sklearn.preprocessing
from ann_benchmarks.algorithms.base import BaseANN
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility



class Milvus(BaseANN):
    def __init__(self, metric, index_type, nlist, collection_name='ann_test'):
        self._nlist = nlist
        self._nprobe = None
        self._metric = metric
        # create connection
        _host = os.getenv('MILVUS2_ENDPOINT')
        connections.connect(host=_host, port='19530')
        self._collection = None
        self._collection_name = collection_name
        self._index_type = index_type
        self._vec_field_name = "embedding"
        self._id_field_name = "id"

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        dim = X.shape[1]
        print(f'dimension: {dim}')
        coll_fields = [
            FieldSchema("id", DataType.INT64, is_primary=True),
            FieldSchema(name=self._vec_field_name, dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        print(coll_fields)
        # create collection
        coll_schema = CollectionSchema(fields=coll_fields, description="ann test collection")
        if utility.has_collection(self._collection_name):
            utility.drop_collection(self._collection_name)
        self._collection = Collection(name=self._collection_name, schema=coll_schema)
        # insert data
        ids = [i for i in range(len(X))]
        self._collection.insert([ids, X.tolist()])
        # create index
        coll_index = {"index_type": self._index_type,
                      "params": {"nlist": self._nlist},
                      "metric_type": self._metric}
        self._collection.create_index(field_name=self._vec_field_name, index_params=coll_index)
        # load the collection into memory
        self._collection.load()


    def set_query_arguments(self, nprobe):
        if nprobe > self._nlist:
            print('nprobe > nlist')
            nprobe = self._nlist
        else:
            self._nprobe = nprobe

    def query(self, v, topK):
        if self._metric == 'angular':
            v /= numpy.linalg.norm(v)
        v = v.tolist()
        search_param = {
            "data": v,
            "anns_field": self._vec_field_name,
            "param": {"metric_type": self._metric, "nprobe": self._nprobe},
            "limit": topK
        }
        results = self._collection.search(**search_param)
        hits = results[0]
        return hits.ids


def __str__(self):
        return 'Milvus(index_type=%s, nlist=%d, nprobe=%d)' % (self._index_type, self._nlist, self._nprobe)
