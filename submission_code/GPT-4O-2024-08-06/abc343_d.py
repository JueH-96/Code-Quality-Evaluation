def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    
    # Initialize scores for each player
    scores = [0] * N
    
    # Read the operations
    operations = []
    index = 2
    for _ in range(T):
        A = int(data[index]) - 1  # Convert to 0-based index
        B = int(data[index + 1])
        operations.append((A, B))
        index += 2
    
    # Process each operation and calculate the number of distinct scores
    distinct_counts = []
    for A, B in operations:
        scores[A] += B
        distinct_scores = len(set(scores))
        distinct_counts.append(distinct_scores)
    
    # Output the result
    for count in distinct_counts:
        print(count)

if __name__ == "__main__":
    main()