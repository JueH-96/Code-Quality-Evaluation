N = int(input())

def carpet(K):
    if K == 0:
        return ["#"]
    prev = carpet(K-1)
    res = []
    for p in prev:
        res.append(p*3)
    for p in prev:
        res.append(p + "."*len(p) + p)
    for p in prev:
        res.append(p*3)
    return res

print("
".join(carpet(N)))