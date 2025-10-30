import collections
import sys

# Use sys.stdin.readline for potentially faster input.
# input() is actually an alias to sys.stdin.readline in this scope.
input = sys.stdin.readline

def main():
    N, D = map(int, input().split())
    
    coords = []
    for _ in range(N):
        X, Y = map(int, input().split())
        coords.append((X, Y))
        
    # Pre-calculate D squared to avoid repeated multiplication or using sqrt.
    # D is an integer, so D*D is exact.
    D_squared = D * D
    
    # infected[i] is True if person i (0-indexed) is infected, False otherwise.
    infected = [False] * N
    
    # Queue for BFS. Stores indices of infected people whose neighbors need processing.
    q = collections.deque()
    
    # Person 1 (0-indexed in our lists) is initially infected.
    # Constraints state N >= 1, so person 0 always exists.
    if N > 0: # Defensive check, though N >= 1 is guaranteed.
        infected[0] = True
        q.append(0)
    else: # N=0 case, though not per constraints.
        return

    # Start BFS
    while q:
        u_idx = q.popleft()  # Current infected person whose neighbors we check
        Xu, Yu = coords[u_idx]
        
        # Check all other people (v_idx) to see if u_idx infects them
        for v_idx in range(N):
            # If v_idx is not already infected
            if not infected[v_idx]:
                Xv, Yv = coords[v_idx]
                
                # Calculate squared Euclidean distance
                # (Xu - Xv)^2 + (Yu - Yv)^2
                # Assign to dx, dy first for clarity, though (Xu-Xv)**2 also works.
                dx = Xu - Xv
                dy = Yu - Yv
                dist_sq = dx*dx + dy*dy
                
                # If person v_idx is within distance D of person u_idx
                if dist_sq <= D_squared:
                    infected[v_idx] = True  # Mark v_idx as infected
                    q.append(v_idx)         # Add v_idx to queue to process its neighbors later
                    
    # Output the results
    # The i-th line of output corresponds to person i+1 in 1-based indexing,
    # or person i in 0-based indexing.
    for i in range(N):
        if infected[i]:
            sys.stdout.write("Yes
")
        else:
            sys.stdout.write("No
")

if __name__ == '__main__':
    main()