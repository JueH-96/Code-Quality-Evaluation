import heapq

def main():
    X, Y, Z = map(int, input().split())
    S = input().strip()
    n = len(S)
    INF = float('inf')
    
    # DP table: dp[i][s] is the minimal cost to type first i characters with state s
    dp = [[INF] * 2 for _ in range(n + 1)]
    dp[0][0] = 0  # initial state: 0 characters, Caps Lock off
    dp[0][1] = Z  # initial state with Caps Lock pressed once
    
    heap = []
    heapq.heappush(heap, (0, 0, 0))
    heapq.heappush(heap, (Z, 0, 1))
    
    while heap:
        cost, i, s = heapq.heappop(heap)
        if cost > dp[i][s]:
            continue
        
        # Action c: press Caps Lock
        new_s = 1 - s
        new_cost = cost + Z
        if new_cost < dp[i][new_s]:
            dp[i][new_s] = new_cost
            heapq.heappush(heap, (new_cost, i, new_s))
        
        if i < n:
            # Action a: press 'a' alone
            c_a = 'a' if s == 0 else 'A'
            if c_a == S[i]:
                new_i = i + 1
                new_cost_a = cost + X
                if new_cost_a < dp[new_i][s]:
                    dp[new_i][s] = new_cost_a
                    heapq.heappush(heap, (new_cost_a, new_i, s))
            
            # Action b: press 'a' and Shift
            c_b = 'A' if s == 0 else 'a'
            if c_b == S[i]:
                new_i = i + 1
                new_cost_b = cost + Y
                if new_cost_b < dp[new_i][s]:
                    dp[new_i][s] = new_cost_b
                    heapq.heappush(heap, (new_cost_b, new_i, s))
    
    print(min(dp[n][0], dp[n][1]))

if __name__ == "__main__":
    main()