from collections import defaultdict
from sys import stdin, stdout

def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    x_root = find(parent, x)
    y_root = find(parent, y)

    if rank[x_root] < rank[y_root]:
        parent[x_root] = y_root
    elif rank[x_root] > rank[y_root]:
        parent[y_root] = x_root
    else:
        parent[y_root] = x_root
        rank[x_root] += 1

def main():
    n, m = map(int, stdin.readline().split())
    parent = list(range(m + 1))
    rank = [0] * (m + 1)

    for _ in range(n):
        a = int(stdin.readline())
        s = list(map(int, stdin.readline().split()))
        for i in range(a - 1):
            union(parent, rank, s[i], s[i + 1])

    if find(parent, 1) == find(parent, m):
        print(0)
    else:
        graph = defaultdict(list)
        for i in range(1, n + 1):
            a = int(stdin.readline())
            s = list(map(int, stdin.readline().split()))
            for j in range(a):
                graph[s[j]].append(i)

        visited = [False] * (n + 1)
        queue = []
        for i in range(1, m + 1):
            if find(parent, i) == find(parent, 1):
                for j in graph[i]:
                    queue.append(j)
                    visited[j] = True

        count = 0
        while queue:
            count += 1
            new_queue = []
            for i in queue:
                for j in range(1, m + 1):
                    if find(parent, j) == find(parent, m):
                        for k in graph[j]:
                            if not visited[k]:
                                new_queue.append(k)
                                visited[k] = True
            queue = new_queue

        if any(find(parent, i) == find(parent, m) for i in range(1, m + 1)):
            print(count)
        else:
            print(-1)

if __name__ == "__main__":
    main()