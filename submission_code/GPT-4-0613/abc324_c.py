N, T = input().strip().split()
N = int(N)
S = [input().strip() for _ in range(N)]

def check(s, t):
    if s == t:
        return True
    if len(s) == len(t):
        diff = sum([s[i] != t[i] for i in range(len(s))])
        if diff == 1:
            return True
    if len(s) == len(t) - 1:
        for i in range(len(s)+1):
            if s[:i] + t[i] + s[i:] == t:
                return True
    if len(s) == len(t) + 1:
        for i in range(len(s)):
            if s[:i] + s[i+1:] == t:
                return True
    return False

res = [i+1 for i in range(N) if check(S[i], T)]
print(len(res))
print(*res)