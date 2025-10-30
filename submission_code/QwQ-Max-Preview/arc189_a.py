MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    if A[0] != 1:
        print(0)
        return
    
    runs = []
    current_val = A[0]
    start = 0
    for i in range(1, N):
        if A[i] != current_val:
            runs.append((current_val, start, i-1))
            current_val = A[i]
            start = i
    runs.append((current_val, start, N-1))
    
    dp = [0] * (N+1)
    dp[0] = 1
    for i in range(1, N+1):
        dp[i] = (2 * dp[i-1]) % MOD
    
    inv = [1] * (N+1)
    for i in range(1, N+1):
        inv[i] = pow(dp[i], MOD-2, MOD)
    
    result = 1
    for val, s, e in runs:
        if val == 1:
            first = s + 1
            if first % 2 == 0:
                result = 0
                break
            last = e + 1
            if last % 2 == 0:
                result = 0
                break
        else:
            first = s + 1
            if first % 2 == 1:
                result = 0
                break
            last = e + 1
            if last % 2 == 1:
                result = 0
                break
        
        initial = []
        for i in range(s, e+1):
            pos = i + 1
            if (pos % 2 == 0 and A[i] == 0) or (pos % 2 == 1 and A[i] == 1):
                initial.append(pos)
        
        m = len(initial)
        if m == 0:
            result = 0
            break
        
        result = result * (dp[m-1] - 1) % MOD
    
    print(result % MOD)

if __name__ == "__main__":
    main()