import heapq
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    
    dp = [0]
    for i in range(n):
        next_dp = [-10**18] * (len(dp) + 1)
        for j in range(len(dp)):
            if next_dp[j+1] < dp[j] + a[i]:
                next_dp[j+1] = dp[j] + a[i]
            if j > 0:
                if next_dp[j-1] < dp[j]:
                    next_dp[j-1] = dp[j]
        while next_dp and next_dp[-1] < -10**18:
            next_dp.pop()
        dp = next_dp
    
    print(max(dp))

if __name__ == '__main__':
    main()