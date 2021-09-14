import dill
import os


def cache_item(obj, path):
    if not os.path.exists(path):
        dill.dump(obj, open(path, "wb+"))
        return obj
    else:
        print("Loading from cache")
        return dill.load(open(path, 'rb'))