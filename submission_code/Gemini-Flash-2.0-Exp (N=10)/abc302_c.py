def solve():
    n, m = map(int, input().split())
    s = [input() for _ in range(n)]

    def diff_count(str1, str2):
        count = 0
        for i in range(m):
            if str1[i] != str2[i]:
                count += 1
        return count

    import itertools
    for perm in itertools.permutations(s):
        valid = True
        for i in range(n - 1):
            if diff_count(perm[i], perm[i+1]) != 1:
                valid = False
                break
        if valid:
            print("Yes")
            return
    print("No")

solve()