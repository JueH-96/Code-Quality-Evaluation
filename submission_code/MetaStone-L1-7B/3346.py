def get_smallest_string(s: str, k: int) -> str:
    n = len(s)
    res = []
    remaining = k
    for i in range(n):
        current_char = s[i]
        for c in range(ord('a'), ord('z') + 1):
            c_char = chr(c)
            d1 = (c - current_char) % 26
            d2 = (current_char - c) % 26
            distance = min(d1, d2)
            if distance <= remaining:
                res.append(c_char)
                remaining -= distance
                break
    return ''.join(res)