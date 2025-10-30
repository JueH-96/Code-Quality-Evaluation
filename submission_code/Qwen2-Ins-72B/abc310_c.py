from collections import defaultdict

def z_function(s):
    n = len(s)
    z = [0] * n
    l, r = 0, 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] - 1 > r:
            l, r = i, i + z[i] - 1
    return z

def solve(N, S_list):
    stick_dict = defaultdict(int)
    for S in S_list:
        z = z_function(S)
        prefix = ''.join([S[i] for i in range(z[1], len(S))])
        suffix = ''.join([S[i] for i in range(0, len(S) - z[1])][::-1])
        stick_dict[(prefix, suffix)] += 1
    return len(stick_dict)

N = int(input())
S_list = [input() for _ in range(N)]
print(solve(N, S_list))