def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    X = int(input_data[1])
    
    A = []
    P = []
    B = []
    Q = []
    idx = 2
    for _ in range(N):
        Ai = int(input_data[idx]); Pi = int(input_data[idx+1])
        Bi = int(input_data[idx+2]); Qi = int(input_data[idx+3])
        idx += 4
        A.append(Ai)
        P.append(Pi)
        B.append(Bi)
        Q.append(Qi)
    
    # We'll do a binary search on the achievable production capacity M.
    # For each M, we compute the minimal total cost to ensure each process i
    # can process at least M units. For process i, the cost to reach M is:
    #    min( P_i * ceil(M / A_i), Q_i * ceil(M / B_i) ).
    # If the sum of costs for all i <= X, then M is feasible; otherwise not.
    
    # A safe upper bound for M is up to 10^9 (since X up to 1e7, A,B up to 100).
    # We'll binary search in [0..10^9].
    
    def can_achieve(M):
        total_cost = 0
        for i in range(N):
            # cost if we use only S_i machines:
            s_units = (M + A[i] - 1) // A[i]  # ceil(M / A[i])
            cost_s = s_units * P[i]
            # cost if we use only T_i machines:
            t_units = (M + B[i] - 1) // B[i]  # ceil(M / B[i])
            cost_t = t_units * Q[i]
            total_cost += min(cost_s, cost_t)
            if total_cost > X:  # no need to continue if already over budget
                return False
        return total_cost <= X
    
    left, right = 0, 10**9
    while left < right:
        mid = (left + right + 1) // 2
        if can_achieve(mid):
            left = mid
        else:
            right = mid - 1
    
    print(left)

# Call main() to execute solution
if __name__ == "__main__":
    main()