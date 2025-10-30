def main():
    # Map each vertex of the pentagon to an integer
    index_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    # Read inputs
    S = input().strip()
    T = input().strip()
    
    # Extract points
    S1, S2 = S[0], S[1]
    T1, T2 = T[0], T[1]
    
    # Convert to indices
    s1_index = index_map[S1]
    s2_index = index_map[S2]
    t1_index = index_map[T1]
    t2_index = index_map[T2]
    
    # Function to determine chord type (side or diagonal) in a regular pentagon
    def chord_type(i, j):
        # The difference in indices mod 5
        d = abs(i - j) % 5
        # Normalize so difference is in {1,2} (because 3 -> 2, 4 -> 1)
        d = min(d, 5 - d)
        # side if d=1, diagonal if d=2
        return d
    
    # Compare chord types
    if chord_type(s1_index, s2_index) == chord_type(t1_index, t2_index):
        print("Yes")
    else:
        print("No")

# Do not forget to call main
main()