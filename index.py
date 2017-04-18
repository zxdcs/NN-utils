def get_index_add(dic, elem):
    """
    Get index of the element. Add it if not exsit.
    :param dic: The dictory saving {element:index}
    :param elem: The element to get index
    :return: index
    """
    if elem not in dic:
        dic[elem] = len(dic)
    return dic[elem]
