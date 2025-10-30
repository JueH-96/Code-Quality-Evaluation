MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    result = 1
    for i in range(1, N+1):
        result = result * (i - A[i-1]) % MOD
    print(result)

if __name__ == "__main__":
    main()