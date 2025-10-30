def main():
    # Map each point to its index on a regular pentagon
    pindex = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    # Read inputs
    s = input().strip()
    t = input().strip()
    
    # Function to get the "distance type" in the pentagon
    # 1 for side-length (neighbors), 2 for diagonal
    def get_distance_type(x, y):
        diff = abs(pindex[x] - pindex[y])
        diff = min(diff, 5 - diff)
        # diff can only be 1, 2, or 3, or 4 here (given x != y).
        # 1 or 4 -> side, 2 or 3 -> diagonal
        return 1 if diff in [1, 4] else 2
    
    s_dist = get_distance_type(s[0], s[1])
    t_dist = get_distance_type(t[0], t[1])
    
    # Compare distance types
    print("Yes" if s_dist == t_dist else "No")

# Do not forget to call the main function
main()