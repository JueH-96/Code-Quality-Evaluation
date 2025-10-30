def maxPalindromesAfterOperations(words):
    from collections import Counter
    count = Counter()
    for word in words:
        for c in word:
            count[c] += 1
    E = sum(v // 2 for v in count.values())
    O = sum(1 for v in count.values() if v % 2 != 0)
    sum_len = sum(len(word) for word in words)
    max_k = min(O, len(words))
    required_even = sum(len(word) - (1 if len(word) % 2 else 0) for word in words)
    if required_even <= E and max_k <= sum_len:
        return min(max_k, sum_len - E)
    else:
        return 0