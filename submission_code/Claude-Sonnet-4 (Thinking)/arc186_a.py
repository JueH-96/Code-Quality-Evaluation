def solve():
    N, Q = map(int, input().split())
    queries = [int(input()) for _ in range(Q)]
    
    def is_feasible(r, c):
        """Check if row sums r and column sums c are feasible using Gale-Ryser theorem"""
        if sum(r) != sum(c):
            return False
        
        # Check bounds
        if any(x < 0 or x > N for x in r) or any(x < 0 or x > N for x in c):
            return False
        
        # Gale-Ryser theorem
        r_sorted = sorted(r, reverse=True)
        c_sorted = sorted(c, reverse=True)
        
        for k in range(1, N + 1):
            left_sum = sum(r_sorted[:k])
            right_sum = sum(min(c_sorted[j], k) for j in range(N))
            if left_sum > right_sum:
                return False
        
        return True
    
    def count_fixed_elements(r, c):
        """Count number of fixed elements for given row/column sums"""
        if not is_feasible(r, c):
            return None
        
        fixed = 0
        for i in range(N):
            for j in range(N):
                # Check if (i,j) is forced to be 1
                r_new = list(r)
                c_new = list(c)
                r_new[i] -= 1
                c_new[j] -= 1
                if not is_feasible(r_new, c_new):
                    fixed += 1
                    continue
                
                # Check if (i,j) is forced to be 0
                r_new = list(r)
                c_new = list(c)
                r_new[i] += 1
                c_new[j] += 1
                if not is_feasible(r_new, c_new):
                    fixed += 1
        
        return fixed
    
    def generate_sequences(total, length, max_val):
        """Generate all sequences of given length that sum to total"""
        if length == 1:
            if 0 <= total <= max_val:
                yield [total]
        else:
            for first in range(min(total, max_val) + 1):
                for rest in generate_sequences(total - first, length - 1, max_val):
                    yield [first] + rest
    
    # Find all possible numbers of fixed elements
    valid_counts = set()
    processed = set()
    
    # Try all possible total sums
    for total in range(N * N + 1):
        # Generate all possible row sum sequences
        for r in generate_sequences(total, N, N):
            r_tuple = tuple(r)
            if r_tuple in processed:
                continue
                
            # Generate all possible column sum sequences with same total
            for c in generate_sequences(total, N, N):
                c_tuple = tuple(c)
                if (r_tuple, c_tuple) in processed:
                    continue
                processed.add((r_tuple, c_tuple))
                
                fixed = count_fixed_elements(r, c)
                if fixed is not None:
                    valid_counts.add(fixed)
    
    # Answer queries
    for k in queries:
        if k in valid_counts:
            print("Yes")
        else:
            print("No")

solve()