def main():
    # Points positions in order (We use "A B C D E F G" -> index positions 0-6)
    points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    # Distances between consecutive points: Index: 0->1:3, 1->2:1, 2->3:4, 3->4:1, 4->5:5, 5->6:9
    distances = [3, 1, 4, 1, 5, 9]
    
    import sys
    input_data = sys.stdin.read().split()
    if len(input_data) < 2:
        return
    p, q = input_data[0], input_data[1]
    
    # Convert letters to index positions
    p_index = points.index(p)
    q_index = points.index(q)
    
    # Ensure we iterate from smaller index to larger index
    start = min(p_index, q_index)
    end = max(p_index, q_index)
    
    # Sum the distances from start to end (since distances list indices start at 0 connecting points[0] and points[1])
    total = sum(distances[start:end])
    
    print(total)

if __name__ == '__main__':
    main()