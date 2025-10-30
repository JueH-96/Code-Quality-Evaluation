MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    max_right = [i for i in range(N + 2)]  # 1-based indexing
    
    result = 1
    for i in range(1, N + 1):
        a = A[i - 1]
        s = a + 1
        if s > i:
            s = i
        current_right = max(i, max_right[s])
        result = result * (current_right - s + 1) % MOD
        max_right[s] = current_right
    
    print(result)

if __name__ == "__main__":
    main()