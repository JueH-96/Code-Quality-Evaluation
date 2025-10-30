def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    operations = []
    
    # Create a position map to find where each number is located
    position = {value: index for index, value in enumerate(A)}
    
    for i in range(N):
        # The correct value at index i should be i + 1
        correct_value = i + 1
        
        # If the current value is not the correct one, we need to swap
        if A[i] != correct_value:
            # Find the index of the correct value
            correct_index = position[correct_value]
            
            # Perform the swap in the array
            A[i], A[correct_index] = A[correct_index], A[i]
            
            # Update the positions in the map
            position[A[correct_index]] = correct_index
            position[A[i]] = i
            
            # Record the operation (1-based index)
            operations.append((i + 1, correct_index + 1))
    
    # Output the result
    print(len(operations))
    for op in operations:
        print(op[0], op[1])