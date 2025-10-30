def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val > 0:
        return 1
    elif val < 0:
        return -1
    else:
        return 0

def segments_intersect(p1, p2, p3, p4):
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)
    return (o1 * o2 < 0) and (o3 * o4 < 0)

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    P = []
    for _ in range(N):
        x = int(input[idx])
        y = int(input[idx + 1])
        P.append((x, y))
        idx += 2
    Q = []
    for _ in range(N):
        x = int(input[idx])
        y = int(input[idx + 1])
        Q.append((x, y))
        idx += 2
    
    edges = []
    for i in range(N):
        for j in range(N):
            edges.append((i, j))
    
    # Sort edges based on specific criteria
    edges.sort(key=lambda x: (
        max(P[x[0]][0], Q[x[1]][0]), 
        min(P[x[0]][0], Q[x[1]][0]), 
        P[x[0]][0], 
        P[x[0]][1], 
        Q[x[1]][0], 
        Q[x[1]][1]
    ))
    
    available_P = [True] * N
    available_Q = [True] * N
    selected_edges = []
    
    for i, j in edges:
        if available_P[i] and available_Q[j]:
            valid = True
            for (i_prev, j_prev) in selected_edges:
                p1, p2 = P[i], Q[j]
                p3, p4 = P[i_prev], Q[j_prev]
                if segments_intersect(p1, p2, p3, p4):
                    valid = False
                    break
            if valid:
                selected_edges.append((i, j))
                available_P[i] = False
                available_Q[j] = False
    
    if len(selected_edges) != N:
        print(-1)
    else:
        R = [0] * N
        for (i, j) in selected_edges:
            R[i] = j + 1
        print(' '.join(map(str, R)))

if __name__ == "__main__":
    main()