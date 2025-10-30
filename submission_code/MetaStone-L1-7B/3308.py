from collections import defaultdict

def lastNonEmptyString(s: str) -> str:
    count = defaultdict(int)
    for c in s:
        count[c] += 1
    k = max(count.values()) if count else 0
    result = []
    for c in s:
        count[c] += 1
        if count[c] > (k - 1):
            result.append(c)
    return ''.join(result)