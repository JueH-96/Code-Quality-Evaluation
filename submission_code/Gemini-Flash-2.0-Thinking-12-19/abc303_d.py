import sys

def solve():
    x, y, z = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()
    n = len(s)
    dp = [[0] * 2 for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for caps_lock_on in range(2):
            cost1 = float('inf')
            cost2 = float('inf')
            
            # Operation 1: 'a' key
            if caps_lock_on == 0:
                if s[i] == 'a':
                    cost1 = x + dp[i+1][0]
            else:
                if s[i] == 'A':
                    cost1 = x + dp[i+1][1]
                    
            # Operation 2: 'a' + Shift key
            if caps_lock_on == 0:
                if s[i] == 'A':
                    cost2 = y + dp[i+1][0]
            else:
                if s[i] == 'a':
                    cost2 = y + dp[i+1][1]
                    
            # Operation 3: Caps Lock key
            cost3 = z + dp[i][1 - caps_lock_on]
            
            dp[i][caps_lock_on] = min(cost1, cost2, cost3)
            
    print(dp[0][0])

if __name__ == '__main__':
    solve()