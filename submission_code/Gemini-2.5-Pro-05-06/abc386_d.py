import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    constraints_input = []
    for _ in range(M):
        r_str, c_str, char_val = sys.stdin.readline().split()
        constraints_input.append((int(r_str), int(c_str), char_val))

    min_L_for_row = {} 
    max_L_for_row = {} 
    
    row_coords_set = set()
    for r, c, char_val in constraints_input:
        row_coords_set.add(r)

    for r_coord in row_coords_set:
        min_L_for_row[r_coord] = 0
        max_L_for_row[r_coord] = N
        
    for r, c, char_val in constraints_input:
        if char_val == 'B':
            min_L_for_row[r] = max(min_L_for_row[r], c)
        else: # char_val == 'W'
            max_L_for_row[r] = min(max_L_for_row[r], c - 1)
            
    for r_coord in row_coords_set:
        if min_L_for_row[r_coord] > max_L_for_row[r_coord]:
            print("No")
            return
            
    sorted_row_coords = sorted(list(row_coords_set))
    
    if sorted_row_coords: 
        # Propagate L_r <= L_{r-1} (max_L_for_row[r] constrained by max_L_for_row[r-1])
        for i in range(1, len(sorted_row_coords)):
            r_curr = sorted_row_coords[i]
            r_prev = sorted_row_coords[i-1]
            max_L_for_row[r_curr] = min(max_L_for_row[r_curr], max_L_for_row[r_prev])

        # Propagate L_r >= L_{r+1} (min_L_for_row[r] constrained by min_L_for_row[r+1])
        for i in range(len(sorted_row_coords) - 2, -1, -1): 
            r_curr = sorted_row_coords[i]
            r_next = sorted_row_coords[i+1]
            min_L_for_row[r_curr] = max(min_L_for_row[r_curr], min_L_for_row[r_next])
            
    for r_coord in row_coords_set: 
        if min_L_for_row[r_coord] > max_L_for_row[r_coord]:
            print("No")
            return
            
    print("Yes")

solve()