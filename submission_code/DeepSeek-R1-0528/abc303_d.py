import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    first_line = data[0].split()
    X = int(first_line[0])
    Y = int(first_line[1])
    Z = int(first_line[2])
    S = data[1].strip()
    
    INF = 10**18
    dp = [0, INF]
    
    for c in S:
        new_dp = [INF, INF]
        for s in range(2):
            if dp[s] == INF:
                continue
            for t in range(2):
                cost0 = 0
                if s != 0:
                    cost0 += Z
                if c == 'a':
                    cost0 += X
                else:
                    cost0 += Y
                if t != 0:
                    cost0 += Z
                    
                cost1 = 0
                if s != 1:
                    cost1 += Z
                if c == 'A':
                    cost1 += X
                else:
                    cost1 += Y
                if t != 1:
                    cost1 += Z
                    
                total_cost = min(cost0, cost1)
                if dp[s] + total_cost < new_dp[t]:
                    new_dp[t] = dp[s] + total_cost
                    
        dp = new_dp
        
    print(min(dp))

if __name__ == "__main__":
    main()