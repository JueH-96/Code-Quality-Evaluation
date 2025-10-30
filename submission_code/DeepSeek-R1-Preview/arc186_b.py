MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    result = 1
    for i in range(N):
        # Convert to 1-based index
        idx = i + 1
        a = A[i]
        term = idx - a
        result = (result * term) % MOD
    print(result)

if __name__ == '__main__':
    main()