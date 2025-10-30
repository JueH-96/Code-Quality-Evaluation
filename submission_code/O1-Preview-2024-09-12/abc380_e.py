# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, Q = map(int, sys.stdin.readline().split())

    parent = [i for i in range(N+1)]
    rank = [0]*(N+1)
    size = [1]*(N+1)
    color = [i for i in range(N+1)]
    total_size = [0]*(N+1)
    for i in range(1,N+1):
        total_size[color[i]] += 1

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x_root, y_root):
        if rank[x_root] < rank[y_root]:
            parent[x_root] = y_root
            size[y_root] += size[x_root]
        else:
            parent[y_root] = x_root
            size[x_root] += size[y_root]
            if rank[x_root] == rank[y_root]:
                rank[x_root] +=1

    output = []
    for _ in range(Q):
        query = sys.stdin.readline().split()
        if not query:
            continue
        if query[0]=='1':
            _, x_str, c_str = query
            x = int(x_str)
            c = int(c_str)
            x_root = find(x)
            old_color = color[x_root]
            if old_color != c:
                total_size[old_color] -= size[x_root]
                total_size[c] += size[x_root]
                color[x_root] = c

            for y in [x-1, x+1]:
                if 1 <= y <= N:
                    y_root = find(y)
                    if y_root != x_root and color[y_root]==c:
                        union(x_root, y_root)
                        x_root = find(x_root)  # Update x_root after union
                        color[x_root] = c  # Ensure the root has the correct color

        elif query[0]=='2':
            _, c_str = query
            c = int(c_str)
            output.append(str(total_size[c]))
    for line in output:
        print(line)
threading.Thread(target=main).start()