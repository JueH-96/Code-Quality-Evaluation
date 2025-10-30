# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read().strip()
    
    S1S2, T1T2 = input[:2], input[2:]
    
    # Define the pentagon vertices in order
    vertices = ['A', 'B', 'C', 'D', 'E']
    
    # Function to calculate the distance between two points in a regular pentagon
    def distance(p1, p2):
        idx1 = vertices.index(p1)
        idx2 = vertices.index(p2)
        return min(abs(idx1 - idx2), 5 - abs(idx1 - idx2))
    
    # Calculate distances
    dist_S1S2 = distance(S1S2[0], S1S2[1])
    dist_T1T2 = distance(T1T2[0], T1T2[1])
    
    # Compare distances
    if dist_S1S2 == dist_T1T2:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()