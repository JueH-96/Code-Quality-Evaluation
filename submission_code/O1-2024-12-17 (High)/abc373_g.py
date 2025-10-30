def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    # Read P points
    px = [0]*N
    py = [0]*N
    idx = 1
    for i in range(N):
        px[i] = int(input_data[idx]); py[i] = int(input_data[idx+1])
        idx += 2
    
    # Read Q points
    qx = [0]*N
    qy = [0]*N
    for i in range(N):
        qx[i] = int(input_data[idx]); qy[i] = int(input_data[idx+1])
        idx += 2
    
    # R[i] = j means P_i is matched with Q_j (both 0-based).
    R = [-1]*N
    
    # A small helper to partition points (by indices) into "left" or "right"
    # relative to the directed line from P p0 to Q q0.
    def partition(p_list, q_list, p0, q0):
        """
        Returns pL, pR, qL, qR where:
         - pL are those P in p_list to the "left" (orientation>0),
         - pR are those P in p_list to the "right" (orientation<0),
         - qL, qR similarly for Q in q_list (excluding q0 itself).
        """
        pL, pR = [], []
        qL, qR = [], []
        
        # Coordinates for p0, q0
        x0, y0 = px[p0], py[p0]
        xx, yy = qx[q0], qy[q0]
        
        # Cross product sign helper:
        # cross((xx-x0, yy-y0), (x - x0, y - y0)) = (xx - x0)*(y - y0) - (yy - y0)*(x - x0)
        # > 0 => left, < 0 => right
        for p in p_list:
            if p == p0:
                continue
            cx = (xx - x0)*(py[p] - y0) - (yy - y0)*(px[p] - x0)
            if cx > 0:
                pL.append(p)
            else:
                pR.append(p)
        
        for q in q_list:
            if q == q0:
                continue
            cx = (xx - x0)*(qy[q] - y0) - (yy - y0)*(qx[q] - x0)
            if cx > 0:
                qL.append(q)
            else:
                qR.append(q)
        
        return pL, pR, qL, qR
    
    # Recursive function to attempt to build a non-crossing matching.
    # p_list and q_list are lists of indices of remaining unmatched P's and Q's.
    def solve_pq(p_list, q_list):
        if len(p_list) == 0:
            return True
        if len(p_list) == 1:
            # Must match these single points
            R[p_list[0]] = q_list[0]
            return True
        
        # Pick the leftmost P (smallest x, then y) to fix first
        p0 = p_list[0]
        for i in range(1, len(p_list)):
            # compare (px[p_list[i]], py[p_list[i]]) with (px[p0], py[p0])
            if (px[p_list[i]] < px[p0]) or (px[p_list[i]] == px[p0] and py[p_list[i]] < py[p0]):
                p0 = p_list[i]
        
        # Now try matching p0 with each candidate q in q_list
        for q0 in q_list:
            # Partition the remaining sets
            pL, pR, qL, qR = partition(p_list, q_list, p0, q0)
            # We need balanced partitions for a non-crossing matching to be possible
            if len(pL) == len(qL) and len(pR) == len(qR):
                R[p0] = q0
                if solve_pq(pL, qL) and solve_pq(pR, qR):
                    return True
                R[p0] = -1
        
        return False
    
    # Prepare the full lists of indices.
    p_indices = list(range(N))
    q_indices = list(range(N))
    
    # Quick check: must have same length (they do: both size N).
    # Attempt to solve.
    if solve_pq(p_indices, q_indices):
        # We have a matching in R. Print 1-based indices of Q for each P in order P0..P_{N-1}.
        # If P_i was matched with Q_j, we print (j+1).
        # We must output R in the order of i=0..N-1.
        print(" ".join(str(R[i]+1) for i in range(N)))
    else:
        print(-1)