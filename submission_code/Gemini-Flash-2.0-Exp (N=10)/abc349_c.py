def solve():
    s = input()
    t = input()

    n = len(s)
    
    # Check for subsequence of length 3
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if t == s[i].upper() + s[j].upper() + s[k].upper():
                    print("Yes")
                    return

    # Check for subsequence of length 2 with 'X'
    for i in range(n):
        for j in range(i + 1, n):
            if t == s[i].upper() + s[j].upper() + 'X':
                print("Yes")
                return
    
    print("No")

solve()