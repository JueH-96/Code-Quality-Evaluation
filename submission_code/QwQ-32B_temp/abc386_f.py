import sys

def main():
    K = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    len_S = len(S)
    len_T = len(T)
    
    if abs(len_T - len_S) > K:
        print("No")
        return
    
    required_offset = len_T - len_S
    K_val = K
    INF = float('inf')
    size = 2 * K_val + 1
    
    # Initialize previous DP array
    prev_dp = [INF] * size
    max_j_initial = min(K_val, len_T)
    for offset in range(0, max_j_initial + 1):
        if offset <= len_T:
            prev_dp[offset + K_val] = offset
    
    for i in range(1, len_S + 1):
        curr_dp = [INF] * size
        for prev_offset in range(-K_val, K_val + 1):
            idx_prev = prev_offset + K_val
            prev_cost = prev_dp[idx_prev]
            if prev_cost == INF:
                continue
            j_prev = (i - 1) + prev_offset
            if j_prev < 0 or j_prev >= len_T:
                continue
            
            # Match/Replace
            cost = prev_cost + (0 if S[i-1] == T[j_prev] else 1)
            new_offset = prev_offset
            if -K_val <= new_offset <= K_val:
                idx = new_offset + K_val
                if cost < curr_dp[idx]:
                    curr_dp[idx] = cost
            
            # Insert
            cost = prev_cost + 1
            new_offset = prev_offset
            if -K_val <= new_offset <= K_val:
                idx = new_offset + K_val
                if cost < curr_dp[idx]:
                    curr_dp[idx] = cost
            
            # Delete
            cost = prev_cost + 1
            new_offset = prev_offset - 1
            if -K_val <= new_offset <= K_val:
                idx = new_offset + K_val
                if cost < curr_dp[idx]:
                    curr_dp[idx] = cost
        
        prev_dp = curr_dp.copy()
    
    if abs(required_offset) > K_val:
        print("No")
    else:
        idx_required = required_offset + K_val
        if prev_dp[idx_required] <= K_val:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()