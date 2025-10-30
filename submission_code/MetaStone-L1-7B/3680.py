import sys
import math

sys.setrecursionlimit(1 << 25)

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1

def count_components(n, arr, T):
    if T == 0:
        return n
    arr.sort()
    dsu = DSU(n)
    for i in range(n):
        a = arr[i]
        if a > T:
            continue
        for j in range(i + 1, n):
            b = arr[j]
            if a == b:
                continue
            if a * b > T * math.gcd(a, b):
                continue
            dsu.union(i, j)
    parent = set()
    for i in range(n):
        if arr[i] > T:
            parent.add(arr[i])
        else:
            root = dsu.find(i)
            parent.add(arr[root])
    count = 0
    for num in arr:
        if num > T:
            count += 1
        else:
            root = dsu.find(arr.index(num))
            if root not in parent:
                count += 1
    return count

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    arr = list(map(int, input[1:n+1]))
    T = int(input[n+1])
    print(count_components(n, arr, T))

if __name__ == "__main__":
    main()