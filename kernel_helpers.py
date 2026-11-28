G = "GACGTCTAGCATCGATCGAA"


def site():
    return G + "TGG"


def put(sequence, start, fragment):
    return sequence[:start] + fragment + sequence[start + len(fragment):]
