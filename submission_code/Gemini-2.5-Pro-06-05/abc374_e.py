import sys

def solve():
    """
    Solves the Maximum Production Capacity problem.
    """
    
    # Fast I/O
    input = sys.stdin.readline

    def min_cost_for_process(C, A, P, B, Q):
        """
        Calculates the minimum cost to achieve a capacity of at least C for a single process.
        
        The key idea is that for two machine types S and T, one is generally better for bulk production.
        We can compare the cost to produce a large, common multiple of capacity, e.g., A*B.
        Cost for A*B capacity with S-machines: B * P
        Cost for A*B capacity with T-machines: A * Q
        
        If B*P < A*Q, S-machines are "bulk-cheaper". An optimal solution will use a limited number of T-machines
        (specifically, fewer than A), because any group of A T-machines can be profitably replaced by B S-machines.
        
        Symmetrically, if A*Q < B*P, T-machines are "bulk-cheaper", and an optimal solution will use fewer than B S-machines.
        
        This insight drastically reduces the search space for the number of machines of one type,
        allowing for an efficient calculation.
        """
        if C <= 0:
            return 0

        # Calculate the cost using only one type of machine. This serves as an initial
        # upper bound for the minimum cost and handles the base cases.
        s_only_needed = (C + A - 1) // A
        cost_s_only = s_only_needed * P
        
        t_only_needed = (C + B - 1) // B
        cost_t_only = t_only_needed * Q
        
        min_c = min(cost_s_only, cost_t_only)

        if B * P < A * Q:
            # S is bulk-cheaper. The optimal number of T-machines is less than A.
            # We iterate t from 1 to A-1. The t=0 case (S-only) is already handled.
            for t_count in range(1, A):
                cost_t = t_count * Q
                # Optimization: If the cost of T-machines alone exceeds the current best,
                # no better solution can be found with this t_count.
                if cost_t >= min_c:
                    break
                
                rem_C = C - t_count * B
                if rem_C > 0:
                    s_count = (rem_C + A - 1) // A
                    min_c = min(min_c, cost_t + s_count * P)
                else:
                    # Capacity already met; no S-machines needed.
                    min_c = min(min_c, cost_t)
                    # For larger t_count, cost_t will only increase, so we can stop.
                    break 
        else: # A * Q <= B * P
            # T is bulk-cheaper or equal. Optimal number of S-machines is less than B.
            # We iterate s from 1 to B-1. The s=0 case (T-only) is handled.
            for s_count in range(1, B):
                cost_s = s_count * P
                if cost_s >= min_c:
                    break

                rem_C = C - s_count * A
                if rem_C > 0:
                    t_count = (rem_C + B - 1) // B
                    min_c = min(min_c, cost_s + t_count * Q)
                else:
                    min_c = min(min_c, cost_s)
                    break
        return min_c

    def is_achievable(C, N, X, processes):
        """
        Checks if a production capacity of C is achievable within budget X.
        """
        total_cost = 0
        for A, P, B, Q in processes:
            cost_i = min_cost_for_process(C, A, P, B, Q)
            total_cost += cost_i
            # Early exit if budget is already exceeded
            if total_cost > X:
                return False
        return True

    N, X = map(int, input().split())
    processes = [list(map(int, input().split())) for _ in range(N)]

    # The problem asks for the maximum C such that the total cost is <= X.
    # The cost function is monotonic with respect to C. This structure allows for binary search on the answer C.
    
    low = 0
    # A safe upper bound for capacity. A very loose upper bound can be derived from the max possible
    # capacity for a single process: (max_budget / min_cost_per_machine) * max_capacity_per_machine.
    # E.g., (10^7 / 1) * 100 = 10^9. Let's use a slightly larger value.
    high = 2 * 10**9 
    ans = 0

    while low <= high:
        C = (low + high) // 2
        
        if is_achievable(C, N, X, processes):
            # This capacity is achievable, try for a higher one.
            ans = C
            low = C + 1
        else:
            # This capacity is too expensive, try a lower one.
            high = C - 1
            
    print(ans)

if __name__ == "__main__":
    solve()