import bisect

MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    X = list(map(int, data[1::2]))
    Y = list(map(int, data[2::2]))
    
    # Sort the points by X increasing, then Y increasing
    points = sorted(zip(X, Y), key=lambda x: (x[0], x[1]))
    X_sorted = [x for x, y in points]
    Y_sorted = [y for x, y in points]
    
    # Compress Y coordinates to 1-based index
    Y_list = sorted(list(set(Y_sorted)))
    Y_rank = {y: i+1 for i, y in enumerate(Y_list)}
    Y_sorted_compressed = [Y_rank[y] for y in Y_sorted]
    max_rank = len(Y_list)
    
    # Fenwick Tree implementation
    class FenwickTree:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (size + 2)
        
        def update(self, idx, delta):
            while idx <= self.size:
                self.tree[idx] = (self.tree[idx] + delta) % MOD
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res = (res + self.tree[idx]) % MOD
                idx -= idx & -idx
            return res
    
    ft = FenwickTree(max_rank)
    answer = 0
    # We start with the empty subset, which is handled by the initial query
    # So we add a dummy element at Y=0 (compressed as 0)
    ft.update(0, 1)
    
    for y in Y_sorted_compressed:
        # The number of subsets where this point is the maximum Y is the sum from Y=0 to y-1
        # Because adding this point to all subsets that end with Y < y, plus the subset containing only this point
        # Query sum up to y-1 (since Y_rank starts from 1)
        sum_prev = ft.query(y-1)
        # Update the answer with sum_prev (the new subsets formed by adding this point)
        answer = (answer + sum_prev) % MOD
        # Update the Fenwick Tree at position y with sum_prev
        ft.update(y, sum_prev)
    
    print(answer % MOD)

if __name__ == "__main__":
    main()