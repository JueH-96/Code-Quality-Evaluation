def main():
    import sys
    K = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    n = len(S)
    m = len(T)
    
    if K == 0:
        print("Yes" if S == T else "No")
        return
    
    prev_dp = {}
    max_j = min(K, m)
    for j in range(0, max_j + 1):
        prev_dp[j] = j
    
    for i in range(1, n + 1):
        current_dp = {}
        current_j_min = max(0, i - K)
        current_j_max = min(m, i + K)
        for j in range(current_j_min, current_j_max + 1):
            cost = float('inf')
            # Delete
            if j in prev_dp:
                cost = min(cost, prev_dp[j] + 1)
            # Insert
            if j - 1 >= current_j_min and (j - 1) in current_dp:
                cost = min(cost, current_dp[j - 1] + 1)
            # Replace or match
            if j - 1 >= 0 and (j - 1) in prev_dp:
                add_cost = 0 if S[i - 1] == T[j - 1] else 1
                cost = min(cost, prev_dp[j - 1] + add_cost)
            if cost <= K:
                current_dp[j] = cost
        prev_dp = current_dp
        if not prev_dp:
            break
    
    print("Yes" if m in prev_dp and prev_dp[m] <= K else "No")

if __name__ == "__main__":
    main()