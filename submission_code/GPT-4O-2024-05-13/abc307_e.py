# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    N = int(data[0])
    M = int(data[1])
    MOD = 998244353

    if M == 1:
        print(0)
        return

    # Calculate M * (M-1)^(N-1) % MOD
    result = M * pow(M-1, N-1, MOD) % MOD
    print(result)

if __name__ == "__main__":
    solve()