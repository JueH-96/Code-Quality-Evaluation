def solve():
    s = input()
    t = input()
    n = len(s)
    
    def check_subsequence(sub):
        i = 0
        j = 0
        while i < n and j < len(sub):
            if s[i] == sub[j]:
                j += 1
            i += 1
        return j == len(sub)

    if check_subsequence(t.lower()):
        print("Yes")
        return

    if t[2] == 'X':
        if check_subsequence(t[:2].lower()):
            print("Yes")
            return

    print("No")

solve()