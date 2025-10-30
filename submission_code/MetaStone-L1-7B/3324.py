from collections import Counter

def isPossibleToSplit(lst: List[int]) -> bool:
    cnt = Counter(lst)
    for v in cnt.values():
        if v > 2:
            return False
    m = sum(1 for v in cnt.values() if v == 1)
    t = sum(1 for v in cnt.values() if v == 2)
    k = len(lst) // 2
    a = k - t
    return a >= 0 and a <= m