import sys
import math

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    queries = []
    for i in range(Q):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        C = int(input[ptr])
        ptr += 1
        queries.append((C, L, R, i))  # i is the original index (0-based)
    
    # Check coverage
    covered = [False] * (N + 2)  # 1-based indexing
    for C, L, R, i in queries:
        for j in range(L, R + 1):
            covered[j] = True
    for j in range(1, N + 1):
        if not covered[j]:
            print(-1)
            return
    
    # Check merged intervals
    intervals = []
    for C, L, R, i in queries:
        intervals.append((L, R))
    intervals.sort()
    merged = []
    for L, R in intervals:
        if not merged:
            merged.append((L, R))
        else:
            prev_L, prev_R = merged[-1]
            if L <= prev_R + 1:
                merged[-1] = (prev_L, max(prev_R, R))
            else:
                merged.append((L, R))
    if merged[0][0] != 1 or merged[-1][1] != N or len(merged) != 1:
        print(-1)
        return
    
    # Sort queries by C_i
    queries.sort()
    
    # Initialize Union-Find
    parent = list(range(N + Q + 2))  # Nodes 1..N+Q
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u
    
    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root == v_root:
            return False
        parent[v_root] = u_root
        return True
    
    total_cost = 0
    B = int(math.sqrt(N)) + 1  # Block size
    
    for C, L, R, i_original in queries:
        u = N + 1 + i_original  # New node index
        
        roots = set()
        j = L
        while j <= R:
            block_start = ((j - 1) // B) * B + 1
            block_end = block_start + B - 1
            current_end = min(block_end, R)
            
            # Check if the entire block is within [L, R]
            if block_start >= L and current_end <= R:
                # Full block
                first = block_start
                last = current_end
                first_root = find(first)
                last_root = find(last)
                if first_root == last_root:
                    roots.add(first_root)
                else:
                    # Check each element in the block
                    for k in range(first, last + 1):
                        roots.add(find(k))
            else:
                # Partial block
                for k in range(j, current_end + 1):
                    roots.add(find(k))
            j = current_end + 1
        
        # Union u with all roots in the set
        initial_u_root = find(u)
        for root in roots:
            if find(u) != find(root):
                union(u, root)
                total_cost += C
    
    # Check connectivity
    root = find(1)
    for j in range(1, N + Q + 1):
        if find(j) != root:
            print(-1)
            return
    print(total_cost)

if __name__ == "__main__":
    main()