def solve():
    s = input()
    t = input()
    n = len(s)
    if len(t) == 3:
        # Condition 1: Subsequence of length 3
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if s[i].upper() == t[0] and s[j].upper() == t[1] and s[k].upper() == t[2]:
                        print("Yes")
                        return

        # Condition 2: Subsequence of length 2 + 'X'
        if t[2] == 'X':
            for i in range(n):
                for j in range(i + 1, n):
                    if s[i].upper() == t[0] and s[j].upper() == t[1]:
                        print("Yes")
                        return

    print("No")

if __name__ == "__main__":
    solve()