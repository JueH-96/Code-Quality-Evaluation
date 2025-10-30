# YOUR CODE HERE
def main():
    # Define the positions of the points on the pentagon
    # Assuming the pentagon is labeled A, B, C, D, E in order
    # We can represent the positions as angles in a unit circle
    # For simplicity, let's assign angles to each point
    # A: 0 degrees, B: 72 degrees, C: 144 degrees, D: 216 degrees, E: 288 degrees
    point_to_angle = {'A': 0, 'B': 72, 'C': 144, 'D': 216, 'E': 288}
    
    # Read the input
    S1S2 = input().strip()
    T1T2 = input().strip()
    
    # Extract the points
    S1, S2 = S1S2[0], S1S2[1]
    T1, T2 = T1T2[0], T1T2[1]
    
    # Calculate the angular differences
    angle_S1 = point_to_angle[S1]
    angle_S2 = point_to_angle[S2]
    angle_T1 = point_to_angle[T1]
    angle_T2 = point_to_angle[T2]
    
    # Calculate the smallest angular difference
    diff_S = abs(angle_S1 - angle_S2)
    diff_T = abs(angle_T1 - angle_T2)
    
    # Since the pentagon is symmetric, the distance is the same if the angular difference is the same
    # or if one is the complement to 360 degrees
    if diff_S == diff_T or abs(diff_S - 360) == diff_T or abs(diff_T - 360) == diff_S:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()