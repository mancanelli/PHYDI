from models.phc.fixup_resnet import *
from models.rezero.reznet import *
from models.rezero.phcreznet import *

def GetModel(str_model, quat_data, n, num_classes=10, rezero=False):
    print('Model:', str_model)
    print()

    if str_model == 'resnet18':
        return ResNet18(num_classes=num_classes, rezero=rezero)
    elif str_model == 'fixup_resnet18':
        if quat_data:
            return fixup_resnet18(channels=4, n=n, num_classes=num_classes)
        else:
            return fixup_resnet18(channels=3, n=n, num_classes=num_classes)
    elif str_model == 'phcresnet18':
        if quat_data:
            return PHCResNet18(channels=4, n=n, num_classes=num_classes, rezero=rezero)
        else:
            return PHCResNet18(channels=3, n=n, num_classes=num_classes, rezero=rezero)

    if str_model == 'resnet50':
        return ResNet50(num_classes=num_classes, rezero=rezero)
    elif str_model == 'fixup_resnet50':
        if quat_data:
            return fixup_resnet50(channels=4, n=n, num_classes=num_classes)
        else:
            return fixup_resnet50(channels=3, n=n, num_classes=num_classes)
    elif str_model == 'phcresnet50':
        if quat_data:
            return PHCResNet50(channels=4, n=n, num_classes=num_classes, rezero=rezero)
        else:
            return PHCResNet50(channels=3, n=n, num_classes=num_classes, rezero=rezero)
    
    if str_model == 'resnet152':
        return ResNet152(num_classes=num_classes, rezero=rezero)
    elif str_model == 'fixup_resnet152':
        if quat_data:
            return fixup_resnet152(channels=4, n=n, num_classes=num_classes)
        else:
            return fixup_resnet152(channels=3, n=n, num_classes=num_classes)
    elif str_model == 'phcresnet152':
        if quat_data:
            return PHCResNet152(channels=4, n=n, num_classes=num_classes, rezero=rezero)
        else:
            return PHCResNet152(channels=3, n=n, num_classes=num_classes, rezero=rezero)

    if str_model == 'resnet18large':
        return ResNet18Large(num_classes=num_classes, rezero=rezero)
    elif str_model == 'phcresnet18large':
        if quat_data:
            return PHCResNet18Large(channels=4, n=n, num_classes=num_classes, rezero=rezero)
        else:
            return PHCResNet18Large(channels=3, n=n, num_classes=num_classes, rezero=rezero)

    if str_model == 'resnet50large':
        return ResNet50Large(num_classes=num_classes, rezero=rezero)
    elif str_model == 'phcresnet50large':
        if quat_data:
            return PHCResNet50Large(channels=4, n=n, num_classes=num_classes, rezero=rezero)
        else:
            return PHCResNet50Large(channels=3, n=n, num_classes=num_classes, rezero=rezero)
        
    if str_model == 'resnet152large':
        return ResNet152Large(num_classes=num_classes, rezero=rezero)
    elif str_model == 'phcresnet152large':
        if quat_data:
            return PHCResNet152Large(channels=4, n=n, num_classes=num_classes, rezero=rezero)
        else:
            return PHCResNet152Large(channels=3, n=n, num_classes=num_classes, rezero=rezero)