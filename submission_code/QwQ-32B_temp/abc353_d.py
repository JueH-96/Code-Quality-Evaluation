MOD = 998244353

def main():
    import sys
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    digits = [len(str(x)) for x in A]
    pow10 = [pow(10, d, MOD) for d in digits]
    
    # Compute prefix sums S where S[j] is sum of first j elements (0-based up to j-1)
    S = [0] * (N + 1)
    for i in range(1, N + 1):
        S[i] = (S[i-1] + A[i-1]) % MOD
    
    first_part = 0
    for j in range(N):
        first_part = (first_part + S[j] * pow10[j]) % MOD
    
    second_part = 0
    for j in range(N):
        second_part = (second_part + A[j] * j) % MOD
    
    total = (first_part + second_part) % MOD
    print(total)

if __name__ == "__main__":
    main()