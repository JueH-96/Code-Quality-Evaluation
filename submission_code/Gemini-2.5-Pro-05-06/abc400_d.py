import collections
import sys

def main():
    H, W = map(int, sys.stdin.readline().split())
    A_str, B_str, C_str, D_str = sys.stdin.readline().split()
    # Convert to 0-indexed
    start_r, start_c = int(A_str) - 1, int(B_str) - 1
    end_r, end_c = int(C_str) - 1, int(D_str) - 1

    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # dist[r][c] stores the minimum number of kicks to reach cell (r,c)
    dist = [[float('inf')] * W for _ in range(H)]
    
    # Use a deque for 0-1 BFS
    q = collections.deque()

    # Starting cell
    dist[start_r][start_c] = 0
    q.append((start_r, start_c))

    # Delta row and column for 4 cardinal directions (for walking and kick base direction)
    # Order: Up, Down, Left, Right (can be any consistent order)
    dr_moves = [-1, 1, 0, 0]
    dc_moves = [0, 0, -1, 1]

    while q:
        r, c = q.popleft()
        current_kicks = dist[r][c]

        # Action 1: Move to an adjacent road cell (cost 0 kicks)
        for i in range(4): # Iterate through 4 directions
            nr, nc = r + dr_moves[i], c + dc_moves[i]

            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '.':
                if current_kicks < dist[nr][nc]:
                    dist[nr][nc] = current_kicks
                    q.appendleft((nr, nc)) # Add to front for 0-cost moves
        
        # Action 2: Perform a front kick (cost +1 kick)
        new_kick_cost = current_kicks + 1
        
        # Iterate over 4 kick directions
        for i in range(4): 
            kick_dr, kick_dc = dr_moves[i], dc_moves[i] # Base direction of the kick
            
            # Iterate over cells 1 step and 2 steps away in the kick direction
            for m in range(1, 3): # m is multiplier: 1 for 1 step, 2 for 2 steps
                nr, nc = r + m * kick_dr, c + m * kick_dc
                
                if 0 <= nr < H and 0 <= nc < W:
                    # This cell (nr, nc) becomes reachable via a kick.
                    # It doesn't matter if grid[nr][nc] was originally '.' or '#'.
                    if new_kick_cost < dist[nr][nc]:
                        dist[nr][nc] = new_kick_cost
                        q.append((nr, nc)) # Add to back for 1-cost moves
    
    ans = dist[end_r][end_c]
    
    if ans == float('inf'):
        # This case implies the target is unreachable.
        # Based on problem guarantees (start/end are roads, kicks break walls),
        # a path should generally exist. If not, -1 is typical.
        print("-1") 
    else:
        print(ans)

if __name__ == '__main__':
    main()