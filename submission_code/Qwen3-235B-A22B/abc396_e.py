import sys
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    parent = list(range(N))
    weight = [0] * N

    def find(u):
        path = []
        while parent[u] != u:
            path.append(u)
            u = parent[u]
        root = u
        if not path:
            return (root, 0)
        cum_xor = [0] * (len(path) + 1)
        cum_xor[-1] = 0
        for i in range(len(path)-1, -1, -1):
            cum_xor[i] = weight[path[i]] ^ cum_xor[i+1]
        for i in range(len(path)):
            v = path[i]
            parent[v] = root
            weight[v] = cum_xor[i]
        return (root, cum_xor[0])

    for _ in range(M):
        x, y, z = map(int, sys.stdin.readline().split())
        x -= 1
        y -= 1
        root_x, x_xor = find(x)
        root_y, y_xor = find(y)
        if root_x == root_y:
            if (x_xor ^ y_xor) != z:
                print(-1)
                return
        else:
            parent[root_x] = root_y
            weight[root_x] = x_xor ^ y_xor ^ z
    
    xor_to_root = [0] * N
    component_map = defaultdict(list)
    for u in range(N):
        root, xr = find(u)
        xor_to_root[u] = xr
        component_map[root].append(u)
    
    A = [0] * N
    for root in component_map:
        nodes = component_map[root]
        all_xor = [xor_to_root[u] for u in nodes]
        n = len(all_xor)
        bit_counts = [0]*31
        for k in range(31):
            mask = 1 << k
            cnt = 0
            for x in all_xor:
                if x & mask:
                    cnt +=1
            bit_counts[k] = cnt
        root_val = 0
        for k in reversed(range(31)):
            mask = 1 << k
            cnt = bit_counts[k]
            if cnt < (n - cnt):
                pass
            elif cnt > (n - cnt):
                root_val |= mask
            else:
                pass
        for u in nodes:
            A[u] = root_val ^ xor_to_root[u]
    
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()