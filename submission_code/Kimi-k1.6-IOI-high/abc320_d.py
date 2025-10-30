def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    parent = list(range(N + 1))
    dx = [0] * (N + 1)
    dy = [0] * (N + 1)
    rank_ = [1] * (N + 1)  # Using rank for union by rank
    
    def find(u):
        if parent[u] != u:
            path = []
            while parent[u] != u:
                path.append(u)
                u = parent[u]
            root = u
            for v in reversed(path):
                old_parent = parent[v]
                parent[v] = root
                dx[v] += dx[old_parent]
                dy[v] += dy[old_parent]
        return parent[u]
    
    for _ in range(M):
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        X = int(data[idx])
        idx += 1
        Y = int(data[idx])
        idx += 1
        
        # Find roots of A and B with path compression
        find(A)
        find(B)
        RA = parent[A]
        RB = parent[B]
        
        if RA != RB:
            if RA == 1:
                # Attach RB tree to RA (1)
                parent[RB] = RA
                dx[RB] = dx[A] + X - dx[B]
                dy[RB] = dy[A] + Y - dy[B]
                if rank_[RA] == rank_[RB]:
                    rank_[RA] += 1
            elif RB == 1:
                # Attach RA tree to RB (1)
                parent[RA] = RB
                dx[RA] = dx[B] - X - dx[A]
                dy[RA] = dy[B] - Y - dy[A]
                if rank_[RA] == rank_[RB]:
                    rank_[RB] += 1
            else:
                # Union by rank
                if rank_[RA] > rank_[RB]:
                    # Attach RB to RA
                    parent[RB] = RA
                    dx[RB] = dx[A] + X - dx[B]
                    dy[RB] = dy[A] + Y - dy[B]
                    if rank_[RA] == rank_[RB]:
                        rank_[RA] += 1
                else:
                    # Attach RA to RB
                    parent[RA] = RB
                    dx[RA] = dx[B] - X - dx[A]
                    dy[RA] = dy[B] - Y - dy[A]
                    if rank_[RA] == rank_[RB]:
                        rank_[RB] += 1
    
    # Prepare output
    output = []
    for i in range(1, N + 1):
        root = find(i)
        if root == 1:
            output.append(f"{dx[i]} {dy[i]}")
        else:
            output.append("undecidable")
    
    print('
'.join(output))

if __name__ == "__main__":
    main()