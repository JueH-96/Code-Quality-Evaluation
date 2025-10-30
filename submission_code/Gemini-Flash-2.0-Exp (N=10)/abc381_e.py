def solve():
    n, q = map(int, input().split())
    s = input()
    
    def is_11_22_string(t):
        if len(t) % 2 == 0:
            return False
        mid = (len(t) + 1) // 2
        for i in range(mid - 1):
            if t[i] != '1':
                return False
        if t[mid - 1] != '/':
            return False
        for i in range(mid, len(t)):
            if t[i] != '2':
                return False
        return True

    def find_max_11_22_subsequence_length(t):
        max_len = 0
        for i in range(1 << len(t)):
            subsequence = ""
            for j in range(len(t)):
                if (i >> j) & 1:
                    subsequence += t[j]
            if is_11_22_string(subsequence):
                max_len = max(max_len, len(subsequence))
        return max_len

    for _ in range(q):
        l, r = map(int, input().split())
        sub = s[l-1:r]
        print(find_max_11_22_subsequence_length(sub))

solve()