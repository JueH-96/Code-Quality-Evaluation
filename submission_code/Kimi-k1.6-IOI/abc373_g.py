def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    
    orig_P = []
    for _ in range(N):
        x = int(input[idx])
        y = int(input[idx+1])
        orig_P.append((x, y))
        idx += 2
    
    orig_Q = []
    for _ in range(N):
        x = int(input[idx])
        y = int(input[idx+1])
        orig_Q.append((x, y))
        idx += 2
    
    # Sort P and Q by x then y coordinates, keeping track of original indices
    sorted_P = sorted([(x, y, i) for i, (x, y) in enumerate(orig_P)], key=lambda x: (x[0], x[1]))
    sorted_Q = sorted([(x, y, i) for i, (x, y) in enumerate(orig_Q)], key=lambda x: (x[0], x[1]))
    
    available_Q = [True] * N
    R = [0] * N
    edges = []  # List of tuples of coordinates ((x1,y1), (x2,y2))
    
    for p in sorted_P:
        px, py, p_idx = p
        found = False
        for q in sorted_Q:
            qx, qy, q_idx_q = q
            if available_Q[q_idx_q]:
                # Check if this Q is available and the edge doesn't intersect any existing edges
                valid = True
                for edge in edges:
                    ax, ay = edge[0]
                    bx, by = edge[1]
                    # Compute orientations
                    # For segment AB (px,py)-(qx,qy) and CD (ax,ay)-(bx,by)
                    # Check if AB and CD intersect
                    # Orientation of A, B, C
                    o1 = (qx - px) * (ay - py) - (qy - py) * (ax - px)
                    # Orientation of A, B, D
                    o2 = (qx - px) * (by - py) - (qy - py) * (bx - px)
                    # Orientation of C, D, A
                    o3 = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
                    # Orientation of C, D, B
                    o4 = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
                    
                    if (o1 * o2 < 0) and (o3 * o4 < 0):
                        valid = False
                        break
                if valid:
                    available_Q[q_idx_q] = False
                    R[p_idx] = q_idx_q + 1  # Convert to 1-based index
                    edges.append(((px, py), (qx, qy)))
                    found = True
                    break
        if not found:
            print(-1)
            return
    
    print(' '.join(map(str, R)))

if __name__ == "__main__":
    main()