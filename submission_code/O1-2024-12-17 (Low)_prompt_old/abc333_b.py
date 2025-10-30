def solve():
    # Map each vertex to an integer 0..4
    idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    S = input().strip()
    T = input().strip()
    
    # Extract the points from the input
    S1, S2 = S[0], S[1]
    T1, T2 = T[0], T[1]
    
    # Function to get the "distance type" in a regular pentagon.
    # We'll categorize distances: 
    # distance == 1 or 4 => "side"
    # distance == 2 or 3 => "diagonal"
    def distance_type(a, b):
        d = abs(idx[a] - idx[b])
        d = min(d, 5 - d)  # the smaller wrap-around distance
        return 'side' if d == 1 else 'diagonal'
    
    # Compare the distance type of S1-S2 and T1-T2
    if distance_type(S1, S2) == distance_type(T1, T2):
        print("Yes")
    else:
        print("No")

# Call the solve() function
solve()