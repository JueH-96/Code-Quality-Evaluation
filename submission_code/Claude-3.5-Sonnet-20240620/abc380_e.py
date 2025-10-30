# YOUR CODE HERE
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            if self.size[px] < self.size[py]:
                px, py = py, px
            self.parent[py] = px
            self.size[px] += self.size[py]

def main():
    N, Q = map(int, input().split())
    
    ds = DisjointSet(N)
    color = list(range(N))
    color_count = [1] * N

    for _ in range(Q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            x, c = query[1] - 1, query[2] - 1
            old_color = color[ds.find(x)]
            if old_color != c:
                group_size = ds.size[ds.find(x)]
                color_count[old_color] -= group_size
                color_count[c] += group_size
                color[ds.find(x)] = c
            
            if x > 0 and color[ds.find(x-1)] == c:
                ds.union(x, x-1)
            if x < N-1 and color[ds.find(x+1)] == c:
                ds.union(x, x+1)
        
        else:  # query[0] == 2
            c = query[1] - 1
            print(color_count[c])

if __name__ == "__main__":
    main()