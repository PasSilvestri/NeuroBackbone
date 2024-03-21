

class BackboneFunction:
    def __init__(self, name: str = None, function: callable = None):
        self.name = name if name != None else "Function"
        self.function = function if function != None else lambda *args, **kwargs: None

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)