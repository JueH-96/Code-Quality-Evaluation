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
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # Create sorted unique list
    sorted_A = sorted(A)
    sorted_unique = []
    prev = None
    for num in sorted_A:
        if num != prev:
            sorted_unique.append(num)
            prev = num
    M = len(sorted_unique)
    
    # Initialize Fenwick Trees for count and sum
    fenwick_count = FenwickTree(M)
    fenwick_sum = FenwickTree(M)
    
    total = 0
    for num in A:
        # Find the 0-based rank using bisect_left
        r = bisect.bisect_left(sorted_unique, num)
        # Query the Fenwick Trees up to index r (1-based)
        cnt = fenwick_count.query(r)
        s = fenwick_sum.query(r)
        total += num * cnt - s
        # Update the Fenwick Trees at index r+1 (1-based)
        fenwick_count.update(r + 1, 1)
        fenwick_sum.update(r + 1, num)
    
    print(total)

if __name__ == '__main__':
    main()