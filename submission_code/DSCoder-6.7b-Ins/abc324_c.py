def solve():
    N = int(input().strip())
    T = input().strip()
    S = [input().strip() for _ in range(N)]

    def check(s, t):
        if s == t:
            return True
        if len(s) + 1 == len(t) and all(s[i] == t[i+1] for i in range(len(s))) and t[-1] == s[-1]:
            return True
        if len(s) - 1 == len(t) and all(s[i+1] == t[i] for i in range(len(s)-1)) and s[0] == t[0]:
            return True
        if len(s) == len(t) and sum(x != y for x, y in zip(s, t)) == 1:
            return True
        return False

    res = [i+1 for i in range(N) if check(S[i], T)]
    print(len(res))
    print(' '.join(map(str, res)))

solve()