import bisect

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

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    A = list(map(int, data[1:n+1]))
    
    # Coordinate compression
    sorted_A = sorted(A)
    unique_sorted = []
    prev = None
    for x in sorted_A:
        if x != prev:
            unique_sorted.append(x)
            prev = x
    max_rank = len(unique_sorted)
    
    # Initialize Fenwick Trees
    count_tree = FenwickTree(max_rank)
    sum_tree = FenwickTree(max_rank)
    total = 0
    
    for x in reversed(A):
        # Find the rank of x
        r = bisect.bisect_left(unique_sorted, x) + 1
        # Calculate count and sum of elements > x
        cnt = count_tree.query(max_rank) - count_tree.query(r)
        s = sum_tree.query(max_rank) - sum_tree.query(r)
        total += s - x * cnt
        # Update the trees
        count_tree.update(r, 1)
        sum_tree.update(r, x)
    
    print(total)

if __name__ == "__main__":
    main()