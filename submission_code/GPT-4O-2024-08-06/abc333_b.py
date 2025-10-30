# YOUR CODE HERE
def calculate_distance(p1, p2):
    # Define the order of vertices in the pentagon
    vertices = ['A', 'B', 'C', 'D', 'E']
    
    # Find the indices of the two points
    index1 = vertices.index(p1)
    index2 = vertices.index(p2)
    
    # Calculate the clockwise and counterclockwise distances
    clockwise_distance = (index2 - index1) % 5
    counterclockwise_distance = (index1 - index2) % 5
    
    # Return the minimum of the two distances
    return min(clockwise_distance, counterclockwise_distance)

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    S1S2 = data[0]
    T1T2 = data[1]
    
    S1, S2 = S1S2[0], S1S2[1]
    T1, T2 = T1T2[0], T1T2[1]
    
    # Calculate distances
    distance_S = calculate_distance(S1, S2)
    distance_T = calculate_distance(T1, T2)
    
    # Compare distances
    if distance_S == distance_T:
        print("Yes")
    else:
        print("No")

main()