MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    MOD = 998244353
    
    # Calculate the sum of 1/(s_u) for each node u, where s_u is the size of the subtree rooted at u
    total = 0
    for u in range(1, N * M + 1):
        if u == 0:
            continue
        # Determine the size of the subtree rooted at u
        # The subtree size for row x is M * (N - x + 1)
        x = (u - 1) // N
        s = M * (N - x)
        inv = pow(s, MOD - 2, MOD)
        total = (total + inv) % MOD
    
    print(total)

if __name__ == '__main__':
    main()