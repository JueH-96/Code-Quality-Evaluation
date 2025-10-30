def diff(s1, s2):
    cnt = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            cnt += 1
    return cnt

def solve():
    n, m = map(int, input().split())
    s = [input() for _ in range(n)]
    
    import itertools
    
    for p in itertools.permutations(s):
        ok = True
        for i in range(n - 1):
            if diff(p[i], p[i+1]) != 1:
                ok = False
                break
        if ok:
            print("Yes")
            return
    
    print("No")

solve()