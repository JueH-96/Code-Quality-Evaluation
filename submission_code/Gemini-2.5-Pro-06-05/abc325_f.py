import sys

# It is good practice to increase recursion limit for recursive DP solutions.
sys.setrecursionlimit(200000)

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        line = sys.stdin.readline()
        if not line:
            return
        N = int(line)
        D = list(map(int, sys.stdin.readline().split()))
        L1, C1, K1 = map(int, sys.stdin.readline().split())
        L2, C2, K2 = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Handles potential empty input or parsing errors
        return

    # To optimize DP, we build our state based on the sensor with the smaller K.
    # We will refer to this as 'type-1' and the other as 'type-2' in the code.
    if K1 > K2:
        L1, C1, K1, L2, C2, K2 = L2, C2, K2, L1, C1, K1

    # dp[k]: minimum number of type-2 sensors needed to cover processed sections,
    #        using exactly k type-1 sensors.
    INF = float('inf')
    dp = [INF] * (K1 + 1)
    dp[0] = 0

    for d_i in D:
        # For the current section D_i, calculate the cost in terms of type-2 sensors
        # for each possible number of type-1 sensors used (from 0 to K1).
        cost_k2 = [0] * (K1 + 1)
        for k in range(K1 + 1):
            rem_len = d_i - k * L1
            if rem_len <= 0:
                cost_k2[k] = 0
            else:
                cost_k2[k] = (rem_len + L2 - 1) // L2
        
        new_dp = [INF] * (K1 + 1)

        # The DP transition is a min-plus convolution:
        # new_dp[j] = min_{0 <= k <= j} (dp[k] + cost_k2[j-k])
        # Since dp and cost_k2 arrays are convex, this can be optimized from O(K1^2)
        # to O(K1 * log K1) using a divide-and-conquer approach.
        def convolve_dnc(i_left, i_right, k_left, k_right):
            if i_left > i_right:
                return

            i_mid = (i_left + i_right) // 2
            
            best_k = k_left
            min_val = INF
            
            # The optimal k for i_mid is constrained by [k_left, k_right] and must be <= i_mid.
            limit_k = min(i_mid, k_right)
            for k in range(k_left, limit_k + 1):
                if dp[k] == INF:
                    continue
                
                val = dp[k] + cost_k2[i_mid - k]
                if val < min_val:
                    min_val = val
                    best_k = k
            
            new_dp[i_mid] = min_val
            
            # Because of convexity, the optimal k for indices to the left of i_mid
            # will be less than or equal to best_k.
            convolve_dnc(i_left, i_mid - 1, k_left, best_k)
            # Similarly for indices to the right.
            convolve_dnc(i_mid + 1, i_right, best_k, k_right)

        convolve_dnc(0, K1, 0, K1)
        dp = new_dp

    # After processing all sections, find the minimum total cost.
    min_cost = INF
    for k1 in range(K1 + 1):
        k2 = dp[k1]
        if k2 <= K2:
            cost = k1 * C1 + k2 * C2
            if cost < min_cost:
                min_cost = cost
    
    if min_cost == INF:
        print(-1)
    else:
        print(min_cost)

solve()