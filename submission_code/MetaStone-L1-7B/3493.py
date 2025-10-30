def maxOperations(s: str) -> int:
    total = 0
    current_ones = 0
    for c in s:
        if c == '1':
            current_ones += 1
        else:
            total += current_ones
            current_ones = 0
    return total