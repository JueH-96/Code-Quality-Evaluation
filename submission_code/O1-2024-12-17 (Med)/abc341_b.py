def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:1+N]))
    ST = list(map(int, input_data[1+N:]))

    S = ST[0::2]
    T = ST[1::2]

    # dp[i] will represent the maximum number of units of currency i we can achieve
    # after handling transformations among countries 1..i (in any order).
    # We only need to store the current dp value as we iterate.
    
    dp = A[0]  # dp[1] corresponds to A_1 initially
    for i in range(N - 1):
        # Compute max currency (i+2) based on dp (i+1) and A_(i+2)
        # (Indexing in code is one less than described, so i+1 => i+2 is i => i+1 in code.)
        next_currency = A[i+1] + (dp // S[i]) * T[i]
        dp = next_currency

    # dp now holds the maximum units of currency N
    print(dp)

# Do not forget to call main
if __name__ == "__main__":
    main()