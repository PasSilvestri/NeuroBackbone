
class BackboneFunction:
    def __init__(self, name: str = None, function: callable = None, **kwargs):
        """
        Initializes a function that can be used as loss or as a metric in a NeuroBackbone pipeline.

        Args:
            name (str): The name of the function.
            function (callable): The function to be called.
            **kwargs: Additional parameters to be passed to the function when called.
        """
        self.name = name if name != None else "Function"
        self.function = function if function != None else lambda *args, **kwargs: None
        self.additional_params = kwargs

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs, **self.additional_params)