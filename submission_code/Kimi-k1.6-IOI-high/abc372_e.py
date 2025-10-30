import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    parent = [i for i in range(N + 1)]
    size = [1] * (N + 1)
    top = [[] for _ in range(N + 1)]
    for i in range(N + 1):
        top[i] = [i]
    
    for _ in range(Q):
        while ptr < len(input) and input[ptr] == '':
            ptr += 1
        if input[ptr] == '1':
            # Process type 1 query
            ptr += 1
            u = int(input[ptr])
            ptr += 1
            v = int(input[ptr])
            ptr += 1
            
            # Find roots with path compression
            root_u = u
            while parent[root_u] != root_u:
                parent[root_u] = parent[parent[root_u]]
                root_u = parent[root_u]
            
            root_v = v
            while parent[root_v] != root_v:
                parent[root_v] = parent[parent[root_v]]
                root_v = parent[root_v]
            
            if root_u != root_v:
                # Union by size
                if size[root_u] < size[root_v]:
                    root_u, root_v = root_v, root_u
                parent[root_v] = root_u
                size[root_u] += size[root_v]
                # Merge top lists
                combined = top[root_u] + top[root_v]
                combined.sort(reverse=True)
                top[root_u] = combined[:10]
        else:
            # Process type 2 query
            ptr += 1
            v = int(input[ptr])
            ptr += 1
            k = int(input[ptr])
            ptr += 1
            
            # Find root with path compression
            root = v
            while parent[root] != root:
                parent[root] = parent[parent[root]]
                root = parent[root]
            
            if size[root] < k:
                print(-1)
            else:
                print(top[root][k-1])

if __name__ == "__main__":
    main()