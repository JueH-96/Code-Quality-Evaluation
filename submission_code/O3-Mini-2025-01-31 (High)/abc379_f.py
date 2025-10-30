def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    H = [int(next(it)) for _ in range(n)]
    
    # Read queries in 1-indexed form; convert to 0-indexed.
    queries = []
    for i in range(q):
        l = int(next(it))
        r = int(next(it))
        queries.append((l - 1, r - 1, i))
    
    # For each building j (0-indexed), determine the last building to its left 
    # that is taller than H[j]. Let L[j] be that index, or -1 if none.
    # Then a building j is visible from any building i with i < j if and only if L[j] < i.
    #
    # For a query (l, r), we want to count buildings j with j > r that are visible 
    # from both building l and building r. In our view, j is visible from building i 
    # if L[j] < i. Hence for j to be visible from both l and r (with l < r), it is enough that
    # L[j] < l.
    # So the answer for (l, r) becomes:
    #     count of j in {r+1, r+2, ..., n} with L[j] < l.
    
    L = [0] * n
    stack = []
    for j in range(n):
        # Pop from stack until we find a building taller than H[j]
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if stack:
            L[j] = stack[-1]
        else:
            L[j] = -1
        stack.append(j)
        
    # We need to answer queries: count { j in (r, n-1] such that L[j] < l }.
    # We do this with an offline (sweep‐line) method.
    #
    # Build an array of pairs (L[j], j) sorted by L[j]. Then, process queries in order 
    # of increasing l (converted to 0-indexed) and add all buildings j with L[j] < current query l.
    sorted_buildings = [(L[j], j) for j in range(n)]
    sorted_buildings.sort(key=lambda x: x[0])
    queries.sort(key=lambda x: x[0])
    
    # Fenwick (Binary Indexed) Tree to support point updates and range queries.
    class Fenw:
        __slots__ = ['n', 'fw']
        def __init__(self, n):
            self.n = n
            self.fw = [0] * (n + 1)
        def update(self, i, delta):
            while i <= self.n:
                self.fw[i] += delta
                i += i & -i
        def query(self, i):
            s = 0
            while i:
                s += self.fw[i]
                i -= i & -i
            return s
            
    fenw = Fenw(n)
    res = [0] * q
    ptr = 0
    nb = len(sorted_buildings)
    # Process queries in increasing order of l.
    # For a query (l0, r0, index) we want to count buildings with j > r0 already added.
    for l0, r0, qi in queries:
        while ptr < nb and sorted_buildings[ptr][0] < l0:
            j = sorted_buildings[ptr][1]
            # In our Fenw tree, we store a building at position j+1 (since our tree is 1-indexed).
            fenw.update(j + 1, 1)
            ptr += 1
        total = fenw.query(n)
        upto = fenw.query(r0 + 1)  # counts buildings with index <= r0 (0-indexed).
        res[qi] = total - upto

    sys.stdout.write("
".join(map(str, res)))

if __name__ == '__main__':
    main()