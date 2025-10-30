def solve():
    n, m = map(int, input().split())
    s = [input() for _ in range(n)]

    def diff(s1, s2):
        count = 0
        for i in range(m):
            if s1[i] != s2[i]:
                count += 1
        return count

    import itertools
    for p in itertools.permutations(s):
        valid = True
        for i in range(n - 1):
            if diff(p[i], p[i+1]) != 1:
                valid = False
                break
        if valid:
            print("Yes")
            return
    print("No")

solve()