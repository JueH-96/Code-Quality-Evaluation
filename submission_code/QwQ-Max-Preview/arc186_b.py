MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    result = 1
    for i in range(N):
        # i is 0-based in the list, corresponding to (i+1) in the problem's 1-based index
        a_i = A[i]
        # Compute (i+1 - a_i) since the problem's i is 1-based
        term = (i + 1 - a_i) % MOD
        result = (result * term) % MOD
    print(result)

if __name__ == '__main__':
    main()