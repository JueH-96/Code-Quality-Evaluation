def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    p, q = data[0], data[1]
    
    # Define the points and their respective distances
    points = "ABCDEFG"
    distances = [3, 1, 4, 1, 5, 9]
    
    # Get the indices of the points p and q
    index_p = points.index(p)
    index_q = points.index(q)
    
    # Ensure index_p is less than index_q for easier calculation
    if index_p > index_q:
        index_p, index_q = index_q, index_p
    
    # Calculate the distance between the points
    distance = sum(distances[index_p:index_q])
    
    # Print the result
    print(distance)

if __name__ == "__main__":
    main()