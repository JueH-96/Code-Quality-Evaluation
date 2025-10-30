import collections
import sys

def main():
    N = int(sys.stdin.readline())
    
    # Store Y-coordinates in an array indexed by X-coordinates (0-indexed).
    # y_for_x[x_coord_0idx] stores the 0-indexed Y-coordinate.
    y_for_x = [-1] * N 
    for _ in range(N):
        x_orig, y_orig = map(int, sys.stdin.readline().split())
        # Convert to 0-indexed coordinates
        y_for_x[x_orig - 1] = y_orig - 1 

    # Initial state: tuple of all ball X-coordinates (0 to N-1), which are sorted.
    initial_state_tuple = tuple(range(N))

    q = collections.deque()
    visited = set()

    q.append(initial_state_tuple)
    visited.add(initial_state_tuple)

    MOD = 998244353
    
    while q:
        current_s_tuple = q.popleft()
        
        # Try picking each ball in current_s_tuple as pivot k
        for k_x_coord_0idx in current_s_tuple:
            k_y_coord_0idx = y_for_x[k_x_coord_0idx]

            next_s_list = []
            # For each ball i in current_s_tuple
            for i_x_coord_0idx in current_s_tuple:
                i_y_coord_0idx = y_for_x[i_x_coord_0idx]
                
                removed = False
                # Check removal condition: (X_i < X_k and Y_i < Y_k)
                if i_x_coord_0idx < k_x_coord_0idx:
                    if i_y_coord_0idx < k_y_coord_0idx:
                        removed = True
                # or (X_i > X_k and Y_i > Y_k)
                elif i_x_coord_0idx > k_x_coord_0idx: 
                    if i_y_coord_0idx > k_y_coord_0idx:
                        removed = True
                # If i_x_coord_0idx == k_x_coord_0idx, it's the pivot itself.
                # It is not removed as X_i < X_k and X_i > X_k are both false.
                
                if not removed:
                    next_s_list.append(i_x_coord_0idx)
            
            # next_s_list is already sorted because current_s_tuple was sorted
            # and we iterated through it in order.
            next_s_tuple = tuple(next_s_list)

            if next_s_tuple not in visited:
                visited.add(next_s_tuple)
                q.append(next_s_tuple)
                
    sys.stdout.write(str(len(visited) % MOD) + "
")

if __name__ == '__main__':
    main()