def main():
    import sys
    data = sys.stdin.read().split()
    if not data: 
        return
    it = iter(data)
    n = int(next(it))
    P = [int(next(it)) for _ in range(n)]
    
    # We simulate the insertions in reverse using a Fenwick Tree (Binary Indexed Tree).
    # Think of the final array as being built by inserting i (from 1 to n) at the 
    # P_i-th free position among n indices. When done in forward order this is slow (O(n^2))
    # but we can rebuild the final array by processing in reverse order from i = n down to 1.
    #
    # Explanation:
    # Initially, imagine n positions are free. For i = n, we want to place i in the P_n-th free
    # position from the left. After placing i, we mark that position as occupied.
    # Then, for i = n-1, we find the P_(n-1)-th free position (among the remaining free positions), 
    # and set it to i. Proceeding this way, we populate the final array.
    #
    # To efficiently find the P_i-th free position and update our free positions,
    # we use a Fenwick Tree initialized with ones (each representing a free slot) over n positions.
    
    class Fenw:
        __slots__ = ('n', 'tree')
        def __init__(self, n):
            self.n = n
            self.tree = [0] * (n + 1)
        
        def init(self, arr):
            # Initialize the BIT with arr (1-index based for BIT)
            for i in range(1, self.n + 1):
                self.tree[i] = arr[i - 1]
            for i in range(1, self.n + 1):
                j = i + (i & -i)
                if j <= self.n:
                    self.tree[j] += self.tree[i]
        
        def update(self, i, delta):
            while i <= self.n:
                self.tree[i] += delta
                i += i & -i
        
        def query(self, i):
            s = 0
            while i:
                s += self.tree[i]
                i -= i & -i
            return s
        
        def find_kth(self, k):
            # Find the smallest index i such that the prefix sum >= k.
            idx = 0
            bit = 1 << (self.n.bit_length())
            while bit:
                next_idx = idx + bit
                if next_idx <= self.n and self.tree[next_idx] < k:
                    k -= self.tree[next_idx]
                    idx = next_idx
                bit //= 2
            return idx + 1

    fenw = Fenw(n)
    # all positions are initially free (represented by 1)
    fenw.init([1] * n)
    
    res = [0] * n  # This will store the final array (mapping positions 1..n to values)
    
    # Process from i = n downto 1
    for i in range(n, 0, -1):
        kth_free = P[i - 1]  # because P is given in order, and i's insertion used P_i
        pos = fenw.find_kth(kth_free)  # find the kth free position (1-indexed)
        res[pos - 1] = i              # assign element i at position pos in final array
        fenw.update(pos, -1)          # mark position pos as occupied
    
    sys.stdout.write(" ".join(map(str, res)))

if __name__ == '__main__':
    main()