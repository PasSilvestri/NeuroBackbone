

class BackboneHook():
    """Generic class to represent hooks for a backbone trainer.
    """
    def __init__(self, hook):
        self.__hook = hook
    def __call__(self, *args, **kwargs):
        return self.__hook(*args, **kwargs)
    
#region basic hooks
class EpochStartHook(BackboneHook):
    """Hook called at the beginning of each epoch.
    """
    def __init__(self, hook: callable = lambda epoch, stage: None):
        """
        Initializes the hook function to be called at the beginning of each epoch.

        Args:
            hook (callable): A function that is called at the beginning of each epoch. Default is a lambda function that returns None.
        """
        super().__init__(hook)
        
class PreprocessSamplesHook(BackboneHook):
    """Hook called right before passing the samples to the model.
    """
    def __init__(self, hook: callable = lambda samples, targets, stage: (samples, targets)):
        """
        Initializes the hook function to preprocess samples and targets.

        Args:
            hook (callable): A function that preprocesses samples and targets. Default is a lambda function that returns the samples and targets unchanged.
        """
        super().__init__(hook)

class ProcessOutputHook(BackboneHook):
    """Hook called right after passing the output of the model to the evaluation function.
    """
    def __init__(self, hook: callable = lambda output, stage: output):
        """
        Initializes the hook function to process the output of the model.

        Args:
            hook (callable): A function that processes the output of the model. Default is a lambda function that returns the output unchanged.
        """
        super().__init__(hook)

class NewLossHook(BackboneHook):
    """Hook called right after the model computes a new loss.
    """
    def __init__(self, hook: callable = lambda loss, stage: None):
        """
        Initializes the hook function to be called when a new loss is computed.

        Args:
            hook (callable): A function that is called with the new loss and the stage. Default is a lambda function that returns None.
        """
        super().__init__(hook)

class NewTrainLossHook(NewLossHook):
    """Hook called right after the model computes a new training loss.
    """
    def __init__(self, hook: callable = lambda loss, stage: None):
        """
        Initializes the hook function to be called when a new loss is computed.

        Args:
            hook (callable): A function that is called with the new loss and the stage. Default is a lambda function that returns None.
        """
        superhook = lambda loss, stage: None if stage != 'train' else hook(loss, stage)
        super().__init__(superhook)

class NewValidLossHook(NewLossHook):
    """Hook called right after the model computes a new validation loss.
    """
    def __init__(self, hook: callable = lambda loss, stage: None):
        """
        Initializes the hook function to be called when a new loss is computed.

        Args:
            hook (callable): A function that is called with the new loss and the stage. Default is a lambda function that returns None.
        """
        superhook = lambda loss, stage: None if stage != 'valid' else hook(loss, stage)
        super().__init__(superhook)
        
class NewScoreHook(BackboneHook):
    """Hook called right after the model achieves a new score.
    """
    def __init__(self, hook: callable = lambda score_name, score_val, stage: None):
        """
        Initializes the hook function to be called when a new score is computed.

        Args:
            hook (callable): A function that is called with the new score and the score name. Default is a lambda function that returns None.
        """
        super().__init__(hook)

class NewNamedScoreHook(NewScoreHook):
    """Hook called right after the model achieves a new score value for the specified score name.
    """
    def __init__(self, score_name, hook: callable = lambda score_name, score_val, stage: None):
        """
        Initializes the hook function to be called when a new score value is computed for the given score.

        Args:
            hook (callable): A function that is called with the new score and the score name. Default is a lambda function that returns None.
        """
        self.score_name = score_name
        superhook = lambda score_name, score_val, stage: None if score_name != self.score_name else hook(score_name, score_val, stage)
        super().__init__(superhook)

class NewBestScoreHook(BackboneHook):
    """Hook called right after the model achieves a new best score.
    """
    def __init__(self, hook: callable = lambda score_name, score_val, stage: None):
        """
        Initializes the hook function to be called when a new best score is computed.

        Args:
            hook (callable): A function that is called with the new best score and the score name. Default is a lambda function that returns None.
        """
        super().__init__(hook)

class NewNamedBestScoreHook(NewBestScoreHook):
    """Hook called right after the model achieves a new best score value for the specified score name.
    """
    def __init__(self, score_name, hook: callable = lambda score_name, score_val, stage: None):
        """
        Initializes the hook function to be called when a new best score value is computed for the given score.

        Args:
            hook (callable): A function that is called with the new best score and the score name. Default is a lambda function that returns None.
        """
        self.score_name = score_name
        superhook = lambda score_name, score_val, stage: None if score_name != self.score_name else hook(score_name, score_val, stage)
        super().__init__(superhook)
#endregion        

