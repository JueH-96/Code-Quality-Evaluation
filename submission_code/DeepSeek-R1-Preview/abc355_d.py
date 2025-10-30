import bisect

def main():
    n = int(input())
    intervals = []
    all_values = []
    for _ in range(n):
        l, r = map(int, input().split())
        intervals.append((l, r))
        all_values.append(l)
        all_values.append(r)
    
    # Deduplicate and sort
    all_values = sorted(list(set(all_values)))
    max_val = len(all_values)
    
    # Fenwick Tree implementation
    class FenwickTree:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 2)  # Using 1-based indexing
        
        def update(self, idx, delta=1):
            while idx <= self.n:
                self.tree[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
    
    # Sort intervals by l
    intervals.sort()
    
    ft = FenwickTree(max_val)
    total = 0
    
    for j in range(n):
        l_j, r_j = intervals[j]
        # Find the first index in all_values >= l_j
        k = bisect.bisect_left(all_values, l_j)
        # Calculate count_j
        count_j = ft.query(max_val) - ft.query(k)
        total += count_j
        
        # Insert r_j into Fenwick Tree
        r_rank = bisect.bisect_left(all_values, r_j)
        ft.update(r_rank + 1)  # Fenwick Tree is 1-based
    
    print(total)

if __name__ == "__main__":
    main()