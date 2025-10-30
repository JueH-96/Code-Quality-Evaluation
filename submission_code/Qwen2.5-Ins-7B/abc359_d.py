# YOUR CODE HERE
from collections import defaultdict

def is_good(s, k):
    n = len(s)
    for i in range(n - k + 1):
        if s[i] != '?' and s[i:i+k] == s[i:i+k][::-1]:
            return False
    return True

def count_good_strings(n, k, s):
    mod = 998244353
    dp = defaultdict(int)
    dp[(0, 0, 0)] = 1
    for i in range(n):
        if s[i] == '?':
            new_dp = defaultdict(int)
            for (j, last, count), val in dp.items():
                for c in 'AB':
                    if j + 1 == k and c == last:
                        continue
                    if j + 1 == k and count == 1:
                        continue
                    new_dp[(0, c, 0 if j + 1 == k else 1)] += val
                    new_dp[(0, c, 1)] += val
                    new_dp[(j + 1, c, 0 if j + 1 == k else 1)] += val
            dp = new_dp
        else:
            new_dp = defaultdict(int)
            for (j, last, count), val in dp.items():
                if j + 1 == k and s[i] == last:
                    continue
                if j + 1 == k and count == 1:
                    continue
                new_dp[(0, s[i], 0 if j + 1 == k else 1)] += val
                new_dp[(0, s[i], 1)] += val
                new_dp[(j + 1, s[i], 0 if j + 1 == k else 1)] += val
            dp = new_dp
    return sum(dp.values()) % mod

n, k = map(int, input().split())
s = input().strip()
print(count_good_strings(n, k, s))