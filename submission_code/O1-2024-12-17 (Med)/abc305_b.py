def main():
    # Precompute the distances from A to each point
    # Let's map each point to its position along the line
    letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    
    # Distances between consecutive points: AB=3, BC=1, CD=4, DE=1, EF=5, FG=9
    # Compute cumulative distances from A
    # distance_from_start[i] will store the distance from A to the i-th point
    distance_from_start = [0, 3, 4, 8, 9, 14, 23]  # A=0, B=3, C=4, D=8, E=9, F=14, G=23
    
    p, q = input().split()
    index_p = letter_to_index[p]
    index_q = letter_to_index[q]
    
    # The distance between p and q is the absolute difference of their cumulative distances
    distance = abs(distance_from_start[index_p] - distance_from_start[index_q])
    print(distance)

# Do not remove the line below
main()