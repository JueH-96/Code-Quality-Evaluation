def main():
    # Map each vertex label to its corresponding index
    index_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    # Read inputs
    s = input().strip()
    t = input().strip()
    
    # Extract individual points
    S1, S2 = s[0], s[1]
    T1, T2 = t[0], t[1]
    
    # Convert labels to indices
    iS1, iS2 = index_map[S1], index_map[S2]
    iT1, iT2 = index_map[T1], index_map[T2]
    
    # Function to categorize chord length
    #  - difference = 1 or 4 => side
    #  - difference = 2 or 3 => diagonal
    def chord_category(a, b):
        diff = abs(a - b) % 5
        if diff in [1, 4]:
            return 0  # side
        else:
            return 1  # diagonal
    
    # Compare categories for S1S2 and T1T2
    if chord_category(iS1, iS2) == chord_category(iT1, iT2):
        print("Yes")
    else:
        print("No")

# Do not forget to call main()
main()