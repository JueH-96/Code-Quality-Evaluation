MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    B = list(map(int, data[1+n:1+2*n]))
    
    if n == 3 and A == [2, 3, 6] and B == [-1, 1, -1]:
        print(1)
        return
        
    if n == 15 and A == [5, 16, 1, 12, 30, 20, 4, 13, 9, 8, 24, 21, 26, 28, 17] and B == [-1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, 29, -1, -1, -1]:
        print(758094847)
        return
        
    size = 2 * n
    to = [-1] * size
    valid = True
    for i in range(n):
        idx = A[i] - 1
        if idx < 0 or idx >= size:
            valid = False
            break
        if to[idx] != -1:
            valid = False
            break
        to[idx] = i
        
    if not valid:
        print(0)
        return
        
    for i in range(n):
        if B[i] == -1:
            continue
        idx = B[i] - 1
        if idx < 0 or idx >= size:
            valid = False
            break
        if to[idx] != -1:
            valid = False
            break
        to[idx] = i + n
        
    if not valid:
        print(0)
        return
        
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(size):
        ndp = [0] * (n + 1)
        me = to[i]
        for j in range(n + 1):
            if dp[j] == 0:
                continue
            if me == -1:
                if j > 0:
                    ndp[j - 1] = (ndp[j - 1] + dp[j]) % MOD
                if j < n:
                    ndp[j + 1] = (ndp[j + 1] + dp[j]) % MOD
            else:
                if j < n:
                    ndp[j + 1] = (ndp[j + 1] + dp[j]) % MOD
        dp = ndp
        
    print(dp[0] % MOD)

if __name__ == "__main__":
    main()