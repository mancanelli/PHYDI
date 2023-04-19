
def GetModel(str_model, quat_data, n, num_classes=10, rezero=False):
    print('Model:', str_model)
    print()

    if rezero:
        if str_model == 'resnet20':
            from models.small_reznets.reznet import resnet20
            return resnet20(channels=3, num_classes=num_classes, rezero=rezero)
        elif str_model == 'qresnet20':
            from models.small_reznets.qreznet import qresnet20
            return qresnet20(channels=4, num_classes=num_classes, rezero=rezero)
        elif str_model == 'phcresnet20':
            from models.small_reznets.phcreznet import phcresnet20
            if quat_data:
                return phcresnet20(channels=4, n=n, num_classes=num_classes, rezero=rezero)
            else:
                return phcresnet20(channels=3, n=n, num_classes=num_classes, rezero=rezero)

        if str_model == 'resnet56':
            from models.small_reznets.reznet import resnet56
            return resnet56(channels=3, num_classes=num_classes, rezero=rezero)
        elif str_model == 'qresnet56':
            from models.small_reznets.qreznet import qresnet56
            return qresnet56(channels=4, num_classes=num_classes, rezero=rezero)
        elif str_model == 'phcresnet56':
            from models.small_reznets.phcreznet import phcresnet56
            if quat_data:
                return phcresnet56(channels=4, n=n, num_classes=num_classes, rezero=rezero)
            else:
                return phcresnet56(channels=3, n=n, num_classes=num_classes, rezero=rezero)
        
        if str_model == 'resnet110':
            from models.small_reznets.reznet import resnet110
            return resnet110(channels=3, num_classes=num_classes, rezero=rezero)
        elif str_model == 'qresnet110':
            from models.small_reznets.qreznet import qresnet110
            return qresnet110(channels=4, num_classes=num_classes, rezero=rezero)
        elif str_model == 'phcresnet110':
            from models.small_reznets.phcreznet import phcresnet110
            if quat_data:
                return phcresnet110(channels=4, n=n, num_classes=num_classes, rezero=rezero)
            else:
                return phcresnet110(channels=3, n=n, num_classes=num_classes, rezero=rezero)

        if str_model == 'resnet20large':
            from models.small_reznets.reznet import resnet20large
            return resnet20large(channels=3, num_classes=num_classes, rezero=rezero)
        elif str_model == 'qresnet20large':
            from models.small_reznets.qreznet import qresnet20large
            return qresnet20large(channels=4, num_classes=num_classes, rezero=rezero)
        elif str_model == 'phcresnet20large':
            from models.small_reznets.phcreznet import phcresnet20large
            if quat_data:
                return phcresnet20large(channels=4, n=n, num_classes=num_classes, rezero=rezero)
            else:
                return phcresnet20large(channels=3, n=n, num_classes=num_classes, rezero=rezero)
            
        if str_model == 'resnet110large':
            from models.small_reznets.reznet import resnet110large
            return resnet110large(channels=3, num_classes=num_classes, rezero=rezero)
        elif str_model == 'qresnet110large':
            from models.small_reznets.qreznet import qresnet110large
            return qresnet110large(channels=4, num_classes=num_classes, rezero=rezero)
        elif str_model == 'phcresnet110large':
            from models.small_reznets.phcreznet import phcresnet110large
            if quat_data:
                return phcresnet110large(channels=4, n=n, num_classes=num_classes, rezero=rezero)
            else:
                return phcresnet110large(channels=3, n=n, num_classes=num_classes, rezero=rezero)

    else:
        if str_model == 'resnet20':
            from models.small_resnets.resnet import resnet20
            return resnet20(channels=3, num_classes=num_classes)
        elif str_model == 'qresnet20':
            from models.small_resnets.qresnet import qresnet20
            return qresnet20(channels=4, num_classes=num_classes)
        elif str_model == 'phcresnet20':
            from models.small_resnets.phmresnet import phmresnet20
            if quat_data:
                return phmresnet20(channels=4, n=n, num_classes=num_classes)
            else:
                return phmresnet20(channels=3, n=n, num_classes=num_classes)

        if str_model == 'resnet56':
            from models.small_resnets.resnet import resnet56
            return resnet56(channels=3, num_classes=num_classes)
        elif str_model == 'qresnet56':
            from models.small_resnets.qresnet import qresnet56
            return qresnet56(channels=4, num_classes=num_classes)
        elif str_model == 'phcresnet56':
            from models.small_resnets.phmresnet import phmresnet56
            if quat_data:
                return phmresnet56(channels=4, n=n, num_classes=num_classes)
            else:
                return phmresnet56(channels=3, n=n, num_classes=num_classes)
        
        if str_model == 'resnet110':
            from models.small_resnets.resnet import resnet110
            return resnet110(channels=3, num_classes=num_classes)
        elif str_model == 'qresnet110':
            from models.small_resnets.qresnet import qresnet110
            return qresnet110(channels=4, num_classes=num_classes)
        elif str_model == 'phcresnet110':
            from models.small_resnets.phmresnet import phmresnet110
            if quat_data:
                return phmresnet110(channels=4, n=n, num_classes=num_classes)
            else:
                return phmresnet110(channels=3, n=n, num_classes=num_classes)

        if str_model == 'resnet20large':
            from models.small_resnets.resnet import resnet20large
            return resnet20large(channels=3, num_classes=num_classes)
        elif str_model == 'qresnet20large':
            from models.small_resnets.qresnet import qresnet20large
            return qresnet20large(channels=4, num_classes=num_classes)
        elif str_model == 'phcresnet20large':
            from models.small_resnets.phmresnet import phmresnet20large
            if quat_data:
                return phmresnet20large(channels=4, n=n, num_classes=num_classes)
            else:
                return phmresnet20large(channels=3, n=n, num_classes=num_classes)
            
        if str_model == 'resnet110large':
            from models.small_resnets.resnet import resnet110large
            return resnet110large(channels=3, num_classes=num_classes)
        elif str_model == 'qresnet110large':
            from models.small_resnets.qresnet import qresnet110large
            return qresnet110large(channels=4, num_classes=num_classes)
        elif str_model == 'phcresnet110large':
            from models.small_resnets.phmresnet import phmresnet110large
            if quat_data:
                return phmresnet110large(channels=4, n=n, num_classes=num_classes)
            else:
                return phmresnet110large(channels=3, n=n, num_classes=num_classes)

    ######################################################

    if rezero:
        if str_model == 'resnet18':
            from models.rezero.reznet import ResNet18
            return ResNet18(num_classes=num_classes, rezero=rezero)
        elif str_model == 'qresnet18':
            from models.rezero.qreznet import QResNet18
            return QResNet18(channels=4, num_classes=num_classes, rezero=rezero)
        elif str_model == 'phcresnet18':
            from models.rezero.phcreznet import PHCResNet18
            if quat_data:
                return PHCResNet18(channels=4, n=n, num_classes=num_classes, rezero=rezero)
            else:
                return PHCResNet18(channels=3, n=n, num_classes=num_classes, rezero=rezero)

        if str_model == 'resnet50':
            from models.rezero.reznet import ResNet50
            return ResNet50(num_classes=num_classes, rezero=rezero)
        elif str_model == 'qresnet50':
            from models.rezero.qreznet import QResNet50
            return QResNet50(channels=4, num_classes=num_classes, rezero=rezero)
        elif str_model == 'phcresnet50':
            from models.rezero.phcreznet import PHCResNet50
            if quat_data:
                return PHCResNet50(channels=4, n=n, num_classes=num_classes, rezero=rezero)
            else:
                return PHCResNet50(channels=3, n=n, num_classes=num_classes, rezero=rezero)
        
        if str_model == 'resnet152':
            from models.rezero.reznet import ResNet152
            return ResNet152(num_classes=num_classes, rezero=rezero)
        elif str_model == 'qresnet152':
            from models.rezero.qreznet import QResNet152
            return QResNet152(channels=4, num_classes=num_classes, rezero=rezero)
        elif str_model == 'phcresnet152':
            from models.rezero.phcreznet import PHCResNet152
            if quat_data:
                return PHCResNet152(channels=4, n=n, num_classes=num_classes, rezero=rezero)
            else:
                return PHCResNet152(channels=3, n=n, num_classes=num_classes, rezero=rezero)

        if str_model == 'resnet18large':
            from models.rezero.reznet import ResNet18Large
            return ResNet18Large(num_classes=num_classes, rezero=rezero)
        elif str_model == 'qresnet18large':
            from models.rezero.qreznet import QResNet18Large
            return QResNet18Large(channels=4, num_classes=num_classes, rezero=rezero)
        elif str_model == 'phcresnet18large':
            from models.rezero.phcreznet import PHCResNet18Large
            if quat_data:
                return PHCResNet18Large(channels=4, n=n, num_classes=num_classes, rezero=rezero)
            else:
                return PHCResNet18Large(channels=3, n=n, num_classes=num_classes, rezero=rezero)

        if str_model == 'resnet50large':
            from models.rezero.reznet import ResNet50Large
            return ResNet50Large(num_classes=num_classes, rezero=rezero)
        elif str_model == 'qresnet50large':
            from models.rezero.qreznet import QResNet50Large
            return QResNet50Large(channels=4, num_classes=num_classes, rezero=rezero)
        elif str_model == 'phcresnet50large':
            from models.rezero.phcreznet import PHCResNet50Large
            if quat_data:
                return PHCResNet50Large(channels=4, n=n, num_classes=num_classes, rezero=rezero)
            else:
                return PHCResNet50Large(channels=3, n=n, num_classes=num_classes, rezero=rezero)
            
        if str_model == 'resnet152large':
            from models.rezero.reznet import ResNet152Large
            return ResNet152Large(num_classes=num_classes, rezero=rezero)
        elif str_model == 'qresnet152large':
            from models.rezero.qreznet import QResNet152Large
            return QResNet152Large(channels=4, num_classes=num_classes, rezero=rezero)
        elif str_model == 'phcresnet152large':
            from models.rezero.phcreznet import PHCResNet152Large
            if quat_data:
                return PHCResNet152Large(channels=4, n=n, num_classes=num_classes, rezero=rezero)
            else:
                return PHCResNet152Large(channels=3, n=n, num_classes=num_classes, rezero=rezero)
    
    else:
        if str_model == 'resnet18':
            from models.real.resnet import ResNet18
            return ResNet18(num_classes=num_classes)
        elif str_model == 'qresnet18':
            from models.quat.qresnet import QResNet18
            return QResNet18(channels=4, num_classes=num_classes)
        elif str_model == 'phcresnet18':
            from models.phc.phcresnet import PHCResNet18
            if quat_data:
                return PHCResNet18(channels=4, n=n, num_classes=num_classes)
            else:
                return PHCResNet18(channels=3, n=n, num_classes=num_classes)

        if str_model == 'resnet50':
            from models.real.resnet import ResNet50
            return ResNet50(num_classes=num_classes)
        elif str_model == 'qresnet50':
            from models.quat.qresnet import QResNet50
            return QResNet50(channels=4, num_classes=num_classes)
        elif str_model == 'phcresnet50':
            from models.phc.phcresnet import PHCResNet50
            if quat_data:
                return PHCResNet50(channels=4, n=n, num_classes=num_classes)
            else:
                return PHCResNet50(channels=3, n=n, num_classes=num_classes)
        
        if str_model == 'resnet152':
            from models.real.resnet import ResNet152
            return ResNet152(num_classes=num_classes)
        elif str_model == 'qresnet152':
            from models.quat.qresnet import QResNet152
            return QResNet152(channels=4, num_classes=num_classes)
        elif str_model == 'phcresnet152':
            from models.phc.phcresnet import PHCResNet152
            if quat_data:
                return PHCResNet152(channels=4, n=n, num_classes=num_classes)
            else:
                return PHCResNet152(channels=3, n=n, num_classes=num_classes)

        if str_model == 'resnet18large':
            from models.real.resnet import ResNet18Large
            return ResNet18Large(num_classes=num_classes)
        elif str_model == 'qresnet18large':
            from models.quat.qresnet import QResNet18Large
            return QResNet18Large(channels=4, num_classes=num_classes)
        elif str_model == 'phcresnet18large':
            from models.phc.phcresnet import PHCResNet18Large
            if quat_data:
                return PHCResNet18Large(channels=4, n=n, num_classes=num_classes)
            else:
                return PHCResNet18Large(channels=3, n=n, num_classes=num_classes)

        if str_model == 'resnet50large':
            from models.real.resnet import ResNet50Large
            return ResNet50Large(num_classes=num_classes)
        elif str_model == 'qresnet50large':
            from models.quat.qresnet import QResNet50Large
            return QResNet50Large(channels=4, num_classes=num_classes)
        elif str_model == 'phcresnet50large':
            from models.phc.phcresnet import PHCResNet50Large
            if quat_data:
                return PHCResNet50Large(channels=4, n=n, num_classes=num_classes)
            else:
                return PHCResNet50Large(channels=3, n=n, num_classes=num_classes)
            
        if str_model == 'resnet152large':
            from models.real.resnet import ResNet152Large
            return ResNet152Large(num_classes=num_classes)
        elif str_model == 'qresnet152large':
            from models.quat.qresnet import QResNet152Large
            return QResNet152Large(channels=4, num_classes=num_classes)
        elif str_model == 'phcresnet152large':
            from models.phc.phcresnet import PHCResNet152Large
            if quat_data:
                return PHCResNet152Large(channels=4, n=n, num_classes=num_classes)
            else:
                return PHCResNet152Large(channels=3, n=n, num_classes=num_classes)
