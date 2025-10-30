# YOUR CODE HERE
MOD = 998244353

def main():
    N, M = map(int, input().split())
    if M == 1:
        if N == 1:
            print(1)
        else:
            print(0)
        return
    if N == 1:
        print(M % MOD)
        return
    # Calculate (M-1)^N + (-1)^N * (M-1)
    term1 = pow(M-1, N, MOD)
    term2 = pow(-1, N, MOD) * (M-1)
    total = (term1 + term2) % MOD
    print(total)

if __name__ == "__main__":
    main()