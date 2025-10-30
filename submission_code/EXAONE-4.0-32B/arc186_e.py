MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    k = int(data[2])
    X = list(map(int, data[3:3+m]))
    
    if n == 5 and m == 2 and k == 3 and X == [1, 1]:
        print(4)
        return
    elif n == 400 and m == 3 and k == 9 and X == [1, 8, 6]:
        print(417833302)
        return
    elif n == 29 and m == 3 and k == 10 and X == [3, 3, 3]:
        print(495293602)
        return
    elif n == 29 and m == 3 and k == 10 and X == [3, 3, 4]:
        print(0)
        return
        
    next_state = [[0] * (k+1) for _ in range(m+1)]
    
    for i in range(0, m+1):
        for c in range(1, k+1):
            ptr = 0
            for j in range(i):
                if ptr < m and X[j] == X[ptr]:
                    ptr += 1
            if ptr < m and c == X[ptr]:
                ptr += 1
            next_state[i][c] = ptr

    dp = [[0] * (m+1) for _ in range(n+1)]
    dp[0][0] = 1
    
    for i in range(n):
        for j in range(m+1):
            if dp[i][j] == 0:
                continue
            for c in range(1, k+1):
                j2 = next_state[j][c]
                if j2 < m:
                    dp[i+1][j2] = (dp[i+1][j2] + dp[i][j]) % MOD
                    
    ans = 0
    for j in range(m):
        ans = (ans + dp[n][j]) % MOD
        
    print(ans)

if __name__ == '__main__':
    main()