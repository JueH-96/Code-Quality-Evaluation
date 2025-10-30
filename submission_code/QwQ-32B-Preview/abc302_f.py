class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [0] * size
        self.merge_count = 0  # To count the number of successful merges

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        
        if xroot == yroot:
            return False  # Already in the same set
        
        # Union by rank
        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        else:
            self.parent[yroot] = xroot
            if self.rank[xroot] == self.rank[yroot]:
                self.rank[xroot] += 1
        self.merge_count += 1
        return True

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    idx = 2
    sets = []
    element_to_sets = [[] for _ in range(M + 1)]
    
    for _ in range(N):
        A = int(data[idx])
        idx += 1
        s = list(map(int, data[idx:idx + A]))
        idx += A
        sets.append(s)
        for elem in s:
            element_to_sets[elem].append(_)
    
    uf = UnionFind(N)
    
    for elem in range(1, M + 1):
        sets_containing_elem = element_to_sets[elem]
        for i in range(len(sets_containing_elem)):
            for j in range(i + 1, len(sets_containing_elem)):
                set1 = sets_containing_elem[i]
                set2 = sets_containing_elem[j]
                if uf.union(set1, set2):
                    uf.merge_count += 1
    
    # Find the parents of sets containing 1 and M
    parent1 = uf.find(next(i for i, s in enumerate(sets) if 1 in s))
    parentM = uf.find(next(i for i, s in enumerate(sets) if M in s))
    
    if parent1 == parentM:
        print(uf.merge_count)
    else:
        print(-1)

if __name__ == "__main__":
    main()