def main():
    import sys,sys
    input = sys.stdin.readline
    
    # Fast union-find (DSU) structure that also maintains top k (k <= 10) elements (sorted in descending order)
    # for each connected component.
    class DSU:
        def __init__(self, n):
            # parent pointer for each element (1-indexed)
            self.parent = list(range(n + 1))
            self.rank = [0] * (n + 1)
            # For each component representative, store a list of top vertices (largest first) in that component
            self.top = [[i] for i in range(n + 1)]  # note: index 0 not used
            
            # Also store the size for clarity. Not needed for query because top list is enough.
            self.size = [1] * (n + 1)
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, a, b):
            rootA = self.find(a)
            rootB = self.find(b)
            if rootA == rootB:
                return
            # Union by rank for DSU efficiency.
            if self.rank[rootA] < self.rank[rootB]:
                rootA, rootB = rootB, rootA
            self.parent[rootB] = rootA
            if self.rank[rootA] == self.rank[rootB]:
                self.rank[rootA] += 1
            
            # Merge the top lists - they are stored descendingly.
            merged = merge_top(self.top[rootA], self.top[rootB])
            self.top[rootA] = merged  # merged list is sorted descending order, and only contains up to 10 elements.
            self.size[rootA] += self.size[rootB]
    
    # Given two sorted-descending lists lst1 and lst2 (each of at most length 10), merge them and pick at most 10 largest.
    def merge_top(lst1, lst2):
        i, j = 0, 0
        merged = []
        while (i < len(lst1) or j < len(lst2)) and len(merged) < 10:
            if i < len(lst1) and j < len(lst2):
                if lst1[i] >= lst2[j]:
                    merged.append(lst1[i])
                    i += 1
                else:
                    merged.append(lst2[j])
                    j += 1
            elif i < len(lst1):
                merged.append(lst1[i])
                i += 1
            else:
                merged.append(lst2[j])
                j += 1
        return merged

    # read first line: N and Q
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    
    dsu = DSU(n)
    output_lines = []
    
    for _ in range(q):
        typ = next(it)
        if typ == '1':  # Edge add: 1 u v
            u = int(next(it))
            v = int(next(it))
            dsu.union(u, v)
        elif typ == '2':  # Query: 2 v k => print kth largest in connected component of v 
            v = int(next(it))
            k = int(next(it))
            root = dsu.find(v)
            # top list for this component is stored in descending order.
            if len(dsu.top[root]) < k:
                output_lines.append("-1")
            else:
                output_lines.append(str(dsu.top[root][k - 1]))
    
    sys.stdout.write("
".join(output_lines))
    
if __name__ == '__main__':
    main()