import bisect

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.count_tree = [0] * (self.size + 2)  # 1-based indexing
        self.sum_tree = [0] * (self.size + 2)
    
    def update(self, index, delta_count, delta_sum):
        while index <= self.size:
            self.count_tree[index] += delta_count
            self.sum_tree[index] += delta_sum
            index += index & -index
    
    def query_count(self, index):
        res = 0
        while index > 0:
            res += self.count_tree[index]
            index -= index & -index
        return res
    
    def query_sum(self, index):
        res = 0
        while index > 0:
            res += self.sum_tree[index]
            index -= index & -index
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    queries = []
    Y = []
    for _ in range(Q):
        X = int(input[ptr])
        ptr += 1
        Y_i = int(input[ptr])
        ptr += 1
        queries.append((X, Y_i))
        Y.append(Y_i)
    
    # Collect all unique Y values and zero
    values = set(Y)
    values.add(0)
    sorted_asc = sorted(values)
    M = len(sorted_asc)
    
    # Initialize Fenwick Tree
    fenwick = FenwickTree(M)
    
    # Find rank of zero
    idx_zero = bisect.bisect_left(sorted_asc, 0)
    rank_zero = M - idx_zero
    fenwick.update(rank_zero, N, 0)  # Initialize with all zeros
    
    current_values = [0] * N
    
    for X, Y_i in queries:
        pos = X - 1  # Convert to 0-based index
        old_val = current_values[pos]
        
        # Decrement old_val's count
        idx_old = bisect.bisect_left(sorted_asc, old_val)
        rank_old = M - idx_old
        fenwick.update(rank_old, -1, -old_val)
        
        # Update current value
        current_values[pos] = Y_i
        
        # Increment new_val's count
        idx_new = bisect.bisect_left(sorted_asc, Y_i)
        rank_new = M - idx_new
        fenwick.update(rank_new, 1, Y_i)
        
        # Binary search for the minimal rank where cumulative count >= K
        low, high = 1, M
        ans = M
        while low <= high:
            mid = (low + high) // 2
            cnt = fenwick.query_count(mid)
            if cnt >= K:
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        # Calculate the value at the found rank
        value = sorted_asc[M - ans]
        total = fenwick.query_sum(ans) - (fenwick.query_count(ans) - K) * value
        print(total)

if __name__ == '__main__':
    main()