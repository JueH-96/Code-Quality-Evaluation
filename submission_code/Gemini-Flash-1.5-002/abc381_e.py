def solve():
    n, q = map(int, input().split())
    s = input()
    
    def is_11_22(sub):
        if len(sub) % 2 == 0:
            return False
        mid = (len(sub) + 1) // 2
        for i in range(mid - 1):
            if sub[i] != '1':
                return False
        if sub[mid - 1] != '/':
            return False
        for i in range(mid, len(sub)):
            if sub[i] != '2':
                return False
        return True

    def longest_11_22_subsequence(t):
        max_len = 0
        for i in range(1 << len(t)):
            subsequence = ""
            for j in range(len(t)):
                if (i >> j) & 1:
                    subsequence += t[j]
            if is_11_22(subsequence):
                max_len = max(max_len, len(subsequence))
        return max_len

    for _ in range(q):
        l, r = map(int, input().split())
        substring = s[l-1:r]
        print(longest_11_22_subsequence(substring))

solve()