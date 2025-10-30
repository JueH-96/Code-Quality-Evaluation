import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree_count = [0] * (self.n + 2)  # 1-based indexing
        self.tree_sum = [0] * (self.n + 2)
    
    def update(self, idx, delta_count, delta_sum):
        idx += 1  # Convert to 1-based index
        while idx <= self.n:
            self.tree_count[idx] += delta_count
            self.tree_sum[idx] += delta_sum
            idx += idx & -idx
    
    def query_count(self, idx):
        idx += 1  # Convert to 1-based index
        res = 0
        while idx > 0:
            res += self.tree_count[idx]
            idx -= idx & -idx
        return res
    
    def query_sum(self, idx):
        idx += 1  # Convert to 1-based index
        res = 0
        while idx > 0:
            res += self.tree_sum[idx]
            idx -= idx & -idx
        return res

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    K = int(input[ptr]); ptr +=1
    Q = int(input[ptr]); ptr +=1
    
    queries = []
    y_values = [0]
    for _ in range(Q):
        x = int(input[ptr])-1; ptr +=1
        y = int(input[ptr]); ptr +=1
        queries.append((x, y))
        y_values.append(y)
    
    # Coordinate compression
    sorted_values = sorted(list(set(y_values)), reverse=True)
    value_to_idx = {v:i for i, v in enumerate(sorted_values)}
    M = len(sorted_values)
    ft = FenwickTree(M)
    
    # Initial all zeros
    zero_idx = value_to_idx[0]
    ft.update(zero_idx, N, 0 * N)
    
    A = [0] * N
    
    for x, y in queries:
        old_val = A[x]
        new_val = y
        if old_val == new_val:
            print(ft.query_sum(value_to_idx[sorted_values[0]]) - (ft.query_count(value_to_idx[sorted_values[0]]) - K) * sorted_values[0] if K <= ft.query_count(zero_idx) else 0)
            continue
        
        old_idx = value_to_idx[old_val]
        ft.update(old_idx, -1, -old_val)
        
        new_idx = value_to_idx[new_val]
        ft.update(new_idx, 1, new_val)
        
        A[x] = new_val
        
        # Binary search for the best_idx
        left = 0
        right = M - 1
        best_idx = -1
        while left <= right:
            mid = (left + right) // 2
            cnt = ft.query_count(mid)
            if cnt >= K:
                best_idx = mid
                left = mid + 1
            else:
                right = mid - 1
        
        if best_idx == -1:
            print(0)
            continue
        
        V = sorted_values[best_idx]
        sum_ge_V = ft.query_sum(best_idx)
        count_ge_V = ft.query_count(best_idx)
        sum_required = sum_ge_V - (count_ge_V - K) * V
        print(sum_required)

if __name__ == "__main__":
    main()