def get_label(index):
    if index == "2":
        return "cigarette"
    if index == "6":
        return "rice"
    if index == "11":
        return "vegetable"
    if index == "8":
        return "peel"
    if index == "9":
        return "pulp"
    if index == "23":
        return "can"
    if index == "36":
        return "bottle"
    if index == "37":
        return "battery"
    return "null"


def get_class(imagePath):
    txtPath = imagePath.replace("jpg", "txt")
    with open(txtPath, "r") as f:
        data = f.readline().split(",")
    return get_label(data[1].strip())
