MOD = 998244353

def solve():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    if N == 0:
        print(0)
        return
    
    if A[-1] < 0:
        print(0)
        return
    
    M = N - 1
    B = A[:-1] if M > 0 else []
    
    if M == 0:
        print(1)
        return
    
    class FenwickTree:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 2)  # 1-based indexing
        
        def add(self, idx, delta):
            while idx <= self.n:
                self.tree[idx] = (self.tree[idx] + delta) % MOD
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res = (res + self.tree[idx]) % MOD
                idx -= idx & -idx
            return res
    
    def count_sequences(m, upper_bounds):
        # Initialize left and right arrays
        left = [0] * m
        right = [0] * m
        INF = float('inf')
        for i in range(m-1, -1, -1):
            if i == m-1:
                # Last digit must be zero (handled in upper_bounds)
                left[i] = 0
                right[i] = 0
                continue
            current_max = upper_bounds[i] if upper_bounds[i] != -1 else INF
            current_min = 0
            # Determine the minimal and maximal values for the current position
            # considering future constraints
            min_val = 0
            max_val = current_max if current_max != INF else m + 1
            # Check the next positions' constraints
            next_min = left[i+1] if (i+1) < m else 0
            next_max = right[i+1] if (i+1) < m else INF
            # Compute the allowed range for the current position
            # based on the next positions
            required_min = next_min + 2 if (i+1) < m else 1
            required_max = next_max + 1 if (i+1) < m else INF
            new_min = max(current_min, required_min)
            new_max = min(current_max, required_max)
            if new_min > new_max:
                left[i] = INF
                right[i] = -INF
            else:
                left[i] = new_min
                right[i] = new_max
        
        if left[0] > right[0]:
            return 0
        
        max_possible = m * 2  # A reasonable upper bound for the Fenwick Tree size
        ft = FenwickTree(max_possible)
        
        # Initialize for the first position
        init_min = max(1, left[0])
        init_max = min(B[0], right[0]) if B else right[0]
        if init_min > init_max:
            return 0
        ft.add(init_min, 1)
        if init_min < init_max:
            ft.add(init_max + 1, MOD -1)
        
        for i in range(1, m):
            current_left = left[i]
            current_right = right[i]
            if current_left > current_right:
                return 0
            # Query the sum from (current_left) to (current_right)
            # by first querying up to current_right and subtracting up to current_left -1
            sum_val = (ft.query(current_right) - ft.query(current_left - 1)) % MOD
            if sum_val == 0:
                return 0
            # Update the Fenwick Tree for the next position
            # The next position's allowed values are from current_left to current_right
            # But considering the next step's required sum constraints
            new_min = max(current_left - 1, 0)
            new_max = current_right -1
            ft.add(new_min + 1, sum_val)
            if new_min + 1 < new_max +1:
                ft.add(new_max + 2, MOD - sum_val)
        
        # After processing all steps, the valid sequences must end with a zero
        # So we need to check the sum after m-1 steps equals m-1
        # But according to the earlier transformation, this should accounted for by the upper bounds
        # So the final answer is the sum from 0 to allowed max after m-1 steps
        return ft.query(right[m-1]) % MOD
    
    # The input B is the first M elements of A, which must form a sequence of M elements summing to M
    # However, this approach may need adjustments. For now, using a placeholder.
    upper_bounds = B.copy()
    upper_bounds.append(0)  # the last element must be zero
    res = count_sequences(M + 1, upper_bounds)
    print(res)

solve()