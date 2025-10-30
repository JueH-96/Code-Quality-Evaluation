import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    
    n = int(data[0])
    k = int(data[1])
    A = list(map(int, data[2:2+k]))
    
    if k == 0:
        print(0)
        return
        
    odds = A
    m = k
    dp = [0] * (m + 1)
    if m >= 1:
        dp[1] = 0
        
    for i in range(2, m + 1):
        diff = odds[i - 1] - odds[i - 2]
        if i % 2 == 0:
            dp[i] = dp[i - 2] + diff
        else:
            dp[i] = min(dp[i - 1], dp[i - 2] + diff)
            
    print(dp[m])

if __name__ == "__main__":
    main()