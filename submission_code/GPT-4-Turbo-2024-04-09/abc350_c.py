def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # This will store the operations
    operations = []
    
    # We will use the position array to know where each element is currently located
    position = [0] * (N + 1)
    
    for index in range(N):
        position[A[index]] = index
    
    # We want to sort the array using the minimum number of swaps
    for i in range(1, N + 1):
        while A[i - 1] != i:
            # The current element is not in the right place
            current_value = A[i - 1]
            correct_position = position[i]
            
            # Swap elements
            A[i - 1], A[correct_position] = A[correct_position], A[i - 1]
            # Update the positions
            position[current_value], position[i] = position[i], position[current_value]
            
            # Record the operation
            operations.append((i, correct_position + 1))
    
    # Output the results
    print(len(operations))
    for op in operations:
        print(op[0], op[1])

if __name__ == "__main__":
    main()