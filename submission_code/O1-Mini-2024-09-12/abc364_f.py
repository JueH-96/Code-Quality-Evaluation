# YOUR CODE HERE
import sys
import sys
import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    from sys import stdin
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    operations = []
    idx = 2
    for i in range(1, Q+1):
        L = int(data[idx])
        R = int(data[idx+1])
        C = int(data[idx+2])
        operations.append((C, L, R, N+i))
        idx +=3
    
    operations.sort()
    
    parent = [i for i in range(N + Q +2)]
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return False
        parent[v_root] = u_root
        return True
    
    # Initialize pointers
    ptr = list(range(N + Q +2))
    
    def find_ptr(u):
        if ptr[u] != u:
            ptr[u] = find_ptr(ptr[u])
        return ptr[u]
    
    total_cost = 0
    
    for op in operations:
        C_i, L_i, R_i, node = op
        j = find_ptr(L_i)
        while j <= R_i:
            if union(j, node):
                total_cost += C_i
            temp = j +1
            ptr[j] = find_ptr(temp)
            j = find_ptr(j)
    
    # Check connectivity
    root = find(1)
    connected = True
    for node in range(2, N + Q +1):
        if find(node) != root:
            connected = False
            break
    if connected:
        print(total_cost)
    else:
        print(-1)

if __name__ == "__main__":
    main()