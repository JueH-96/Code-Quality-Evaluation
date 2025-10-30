def main():
    import sys
    data = sys.stdin.read().strip().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    # S_i and T_i pairs start from index 1 + n
    ST = list(map(int, data[1+n:]))

    # Separate S and T for easier access
    S = [ST[2*i] for i in range(n-1)]
    T = [ST[2*i + 1] for i in range(n-1)]

    # c will hold the current amount of the "active" currency
    c = A[0]

    # Convert consecutively from country i to i+1
    for i in range(n - 1):
        # The maximum times we can pay S[i] from c
        times = c // S[i]
        # After conversion we gain T[i] * times in currency i+1
        # plus whatever was initially A[i+1]
        c_next = A[i+1] + T[i] * times
        c = c_next

    print(c)

# Do not forget to call main()
main()