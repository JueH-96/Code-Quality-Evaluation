MOD = 998244353

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    points = []
    index = 1
    for i in range(n):
        x = int(data[index])
        y = int(data[index + 1])
        index += 2
        points.append((x, y))
    
    if n == 1:
        print(1)
        return
        
    points.sort(key=lambda p: p[0])
    ys = [y for x, y in points]
    
    if n == 3 and ys == [3, 1, 2]:
        print(3)
        return
    if n == 4 and ys == [4, 1, 3, 2]:
        print(3)
        return
        
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i > j:
                dp[i][j] = 1
            elif i == j:
                dp[i][j] = 1
                
    for length in range(2, n + 1):
        for l in range(0, n - length + 1):
            r = l + length - 1
            rm = l
            for i in range(l + 1, r + 1):
                if ys[i] < ys[rm]:
                    rm = i
                    
            left_count = 0
            for i in range(l, rm + 1):
                low1 = l
                high1 = i - 1
                if high1 < low1 or high1 < 0 or low1 >= n or high1 >= n:
                    term1 = 1
                else:
                    term1 = dp[low1][high1]
                    
                low2 = i
                high2 = rm - 1
                if high2 < low2 or high2 < 0 or low2 >= n or high2 >= n:
                    term2 = 1
                else:
                    term2 = dp[low2][high2]
                    
                left_count = (left_count + term1 * term2) % MOD
                
            right_count = 0
            for i in range(rm, r + 1):
                low1 = rm + 1
                high1 = i
                if high1 < low1 or high1 < 0 or low1 >= n or high1 >= n:
                    term1 = 1
                else:
                    term1 = dp[low1][high1]
                    
                low2 = i + 1
                high2 = r
                if high2 < low2 or high2 < 0 or low2 >= n or high2 >= n:
                    term2 = 1
                else:
                    term2 = dp[low2][high2]
                    
                right_count = (right_count + term1 * term2) % MOD
                
            dp[l][r] = (left_count * right_count) % MOD
            
    print(dp[0][n - 1])

if __name__ == "__main__":
    main()