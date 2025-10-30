def solve():
    s = input()
    t = input()

    n = len(s)

    # Check for subsequence of length 3
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                subsequence = s[i] + s[j] + s[k]
                uppercase_subsequence = subsequence.upper()
                if uppercase_subsequence == t:
                    print("Yes")
                    return

    # Check for subsequence of length 2
    for i in range(n):
        for j in range(i + 1, n):
            subsequence = s[i] + s[j]
            uppercase_subsequence = subsequence.upper() + "X"
            if uppercase_subsequence == t:
                print("Yes")
                return

    print("No")

solve()