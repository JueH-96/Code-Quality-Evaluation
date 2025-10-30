import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    queries = []
    Ys = []
    for _ in range(Q):
        X = int(input[ptr])
        Y = int(input[ptr + 1])
        queries.append((X, Y))
        Ys.append(Y)
        ptr += 2
    
    # Collect all Y values and initial zeros
    all_values = Ys + [0] * N
    unique_values = sorted(list(set(all_values)), reverse=True)
    M = len(unique_values)
    value_to_rank = {v: i + 1 for i, v in enumerate(unique_values)}  # 1-based ranks
    
    # Fenwick Tree implementation
    class FenwickTree:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 2)  # 1-based indexing
        
        def update(self, idx, delta):
            while idx <= self.n:
                self.tree[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
    
    # Initialize Fenwick Trees for counts and sums
    counts = FenwickTree(M)
    sums = FenwickTree(M)
    
    zero_rank = value_to_rank[0]
    counts.update(zero_rank, N)
    sums.update(zero_rank, 0)
    
    current_values = [0] * (N + 1)  # 1-based indexing for positions
    
    output = []
    for X, Y in queries:
        pos = X
        old_val = current_values[pos]
        new_val = Y
        
        old_rank = value_to_rank[old_val]
        new_rank = value_to_rank[new_val]
        
        # Update counts and sums
        counts.update(old_rank, -1)
        sums.update(old_rank, -old_val)
        counts.update(new_rank, 1)
        sums.update(new_rank, new_val)
        
        current_values[pos] = new_val
        
        # Binary search for the minimal rank i where counts.query(i) >= K
        low, high = 1, M
        ans = M
        while low <= high:
            mid = (low + high) // 2
            cnt = counts.query(mid)
            if cnt >= K:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        # Calculate the sum
        sum_counts = counts.query(ans)
        if ans == 1:
            prev_sum_counts = 0
            prev_sum_sums = 0
        else:
            prev_sum_counts = counts.query(ans - 1)
            prev_sum_sums = sums.query(ans - 1)
        
        needed = K - prev_sum_counts
        value = unique_values[ans - 1]
        sum_result = prev_sum_sums + needed * value
        output.append(str(sum_result))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()