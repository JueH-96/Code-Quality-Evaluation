# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Create a dictionary to map each element to its index
    pos = {x: i for i, x in enumerate(A)}
    
    operations = []
    
    for i in range(N):
        if A[i] != i + 1:
            # Find the current position of the element that should be at position i
            target = i + 1
            current_pos = pos[target]
            
            # Swap A[i] and A[current_pos]
            A[i], A[current_pos] = A[current_pos], A[i]
            
            # Update the positions in the dictionary
            pos[A[i]] = i
            pos[A[current_pos]] = current_pos
            
            # Record the operation
            operations.append((i + 1, current_pos + 1))
    
    # Output the number of operations and the operations themselves
    print(len(operations))
    for op in operations:
        print(op[0], op[1])

if __name__ == "__main__":
    main()