def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    p, q = input_data[0], input_data[1]
    
    # Create a list of points in order and a list of distances between adjacent points:
    points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # distances corresponding to the segment between points[i] and points[i+1]
    distances = [3, 1, 4, 1, 5, 9]
    
    # Find indices for p and q in points list
    i = points.index(p)
    j = points.index(q)
    
    # Ensure i < j by swapping if necessary, because the distance is absolute.
    if i > j:
        i, j = j, i
    
    # Now sum distances from index i to j - 1
    total_distance = sum(distances[i:j])
    sys.stdout.write(str(total_distance))
    
if __name__ == '__main__':
    main()