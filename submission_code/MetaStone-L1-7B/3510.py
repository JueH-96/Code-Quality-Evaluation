def maximumTotalSum(maxH: List[int]) -> int:
    maxH_sorted = sorted(maxH)
    n = len(maxH_sorted)
    last_assigned = 0
    total = 0
    for h in maxH_sorted:
        if h < last_assigned + 1:
            return -1
        total += last_assigned + 1
        last_assigned += 1
    return total