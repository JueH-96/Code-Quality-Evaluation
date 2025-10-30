# YOUR CODE HERE
import sys
import itertools

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx +=1
    M_G = int(data[idx]); idx +=1
    G_edges = []
    for _ in range(M_G):
        u = int(data[idx])-1
        v = int(data[idx+1])-1
        G_edges.append( (u,v) )
        idx +=2
    M_H = int(data[idx]); idx +=1
    H_edges = []
    for _ in range(M_H):
        a = int(data[idx])-1
        b = int(data[idx+1])-1
        H_edges.append( (a,b) )
        idx +=2
    # Read A costs
    A = [[0]*N for _ in range(N)]
    for i in range(N-1):
        for j in range(i+1, N):
            A[i][j] = int(data[idx])
            idx +=1
    # Assign bits to edges
    edge_to_bit = {}
    bit = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_to_bit[(i,j)] = bit
            bit +=1
    # Create G_mask and H_mask
    G_mask = 0
    for (u,v) in G_edges:
        if u > v:
            u, v = v, u
        G_mask |= (1 << edge_to_bit[(u,v)])
    H_mask = 0
    for (a,b) in H_edges:
        if a > b:
            a, b = b, a
        H_mask |= (1 << edge_to_bit[(a,b)])
    min_cost = float('inf')
    for P in itertools.permutations(range(N)):
        cost = 0
        for i in range(N):
            for j in range(i+1, N):
                g_edge = (G_mask >> edge_to_bit[(i,j)]) & 1
                x, y = P[i], P[j]
                if x > y:
                    x, y = y, x
                h_edge = (H_mask >> edge_to_bit[(x,y)]) & 1
                if g_edge != h_edge:
                    cost += A[x][y]
                    if cost >= min_cost:
                        break
            if cost >= min_cost:
                break
        if cost < min_cost:
            min_cost = cost
    print(min_cost)

if __name__ == "__main__":
    main()