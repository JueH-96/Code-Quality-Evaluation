import bisect

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 1)  # 1-based
    
    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index
    
    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def main():
    import sys
    n = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    if n == 0:
        print(0)
        return
    # Coordinate compression
    sorted_unique = sorted(set(A))
    m = len(sorted_unique)
    # Precompute ranks
    ranks = []
    for x in A:
        r = bisect.bisect_left(sorted_unique, x) + 1  # 1-based
        ranks.append(r)
    # Initialize Fenwick Trees
    ft_count = FenwickTree(m)
    ft_sum = FenwickTree(m)
    total = 0
    for x, r in zip(A, ranks):
        # Query up to r-1
        count_less = ft_count.query(r - 1)
        sum_less = ft_sum.query(r - 1)
        contribution = x * count_less - sum_less
        total += contribution
        # Update the trees
        ft_count.update(r, 1)
        ft_sum.update(r, x)
    print(total)

if __name__ == "__main__":
    main()