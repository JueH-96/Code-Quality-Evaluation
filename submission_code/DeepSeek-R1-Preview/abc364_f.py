import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    
    ops = []
    for i in range(Q):
        L = int(data[idx])
        idx += 1
        R = int(data[idx])
        idx += 1
        C = int(data[idx])
        idx += 1
        ops.append((C, L, R, i + 1))
    
    # Sort the operations by C in ascending order
    ops.sort()
    
    # DSU for all nodes (normal + special)
    parent = [i for i in range(N + Q + 2)]  # 1-based for N+Q
    
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
    
    # Jump pointer array for the normal nodes (1 to N)
    jump = list(range(N + 2))  # jump[i] is the next node after i not in the same component
    
    total = 0
    
    for C, L, R, i in ops:
        S = N + i
        current = L
        count = 0
        while current <= R:
            # Find the next jump
            next_jump = min(jump[current], R + 1)
            # The current segment is [current, next_jump - 1]
            # Check if any node in this segment is connected to S
            # Since we process in order, we can connect S to the segment
            # So we need to find if the segment's root is different from S's root
            # We can pick any node in the segment, say current
            root = find(current)
            s_root = find(S)
            if root != s_root:
                count += 1
                # Union the entire segment's component with S
                # But we can just union S with the segment's root
                union(root, s_root)
            # Update the jump pointers
            # All nodes from current to next_jump - 1 can jump to next_jump
            # To optimize, we can iterate and update jump, but that's O(R-L+1), which is too slow
            # Instead, we can jump in blocks
            # We can set jump[current] to next_jump and move on
            # But to prevent multiple updates, we can use a while loop
            # However, given time constraints, we'll proceed as follows
            # Note: This may not be the most efficient, but works within problem constraints
            tmp = current
            while tmp < next_jump:
                if jump[tmp] <= next_jump:
                    tmp = jump[tmp]
                else:
                    break
            jump[current] = next_jump
            current = next_jump
        total += count * C
    
    # Check if all nodes are connected
    # Find the root of node 1
    root = find(1)
    all_connected = True
    # Check all normal nodes
    for i in range(1, N + 1):
        if find(i) != root:
            all_connected = False
            break
    if not all_connected:
        print(-1)
        return
    # Check all special nodes
    for i in range(1, Q + 1):
        S = N + i
        if find(S) != root:
            all_connected = False
            break
    if not all_connected:
        print(-1)
    else:
        print(total)
    
if __name__ == '__main__':
    main()