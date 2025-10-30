import itertools

def max_overlap(x, y):
    max_possible = min(len(x), len(y))
    for k in range(max_possible, 0, -1):
        if x.endswith(y[:k]):
            return k
    return 0

def merge(x, y):
    k = max_overlap(x, y)
    return x + y[k:]

def is_substring(s, substr):
    return any(s[i:i+len(substr)] == substr for i in range(len(s) - len(substr) + 1))

def minimal_string(a, b, c):
    perms = itertools.permutations([a, b, c])
    candidates = []
    for perm in perms:
        x, y, z = perm
        merged = merge(x, y)
        merged = merge(merged, z)
        if is_substring(merged, a) and is_substring(merged, b) and is_substring(merged, c):
            candidates.append(merged)
    if not candidates:
        return ""
    min_len = min(len(s) for s in candidates)
    min_candidates = [s for s in candidates if len(s) == min_len]
    min_candidates.sort()
    return min_candidates[0]