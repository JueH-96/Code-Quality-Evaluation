# YOUR CODE HERE
MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    N = int(input())
    # Calculate the number of digits in N
    num_digits = len(str(N))

    # Calculate the total number of digits in V_N
    total_digits = num_digits * N

    # Calculate the remainder when V_N is divided by 998244353
    remainder = (N * pow(10, total_digits, MOD) - N) % MOD
    remainder = (remainder * pow(10, MOD - 2, MOD)) % MOD
    remainder = (remainder + N) % MOD

    print(remainder)

if __name__ == "__main__":
    main()