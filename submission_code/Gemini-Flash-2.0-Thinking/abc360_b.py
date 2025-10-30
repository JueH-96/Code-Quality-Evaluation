def solve():
    S, T = input().split()
    n = len(S)
    m = len(T)

    for w in range(1, n):
        for c in range(1, w + 1):
            concatenated_string = ""
            start = 0
            while start < n:
                end = min(start + w, n)
                substring = S[start:end]
                if len(substring) >= c:
                    concatenated_string += substring[c-1]
                start += w

            if concatenated_string == T:
                print("Yes")
                return

    print("No")

solve()