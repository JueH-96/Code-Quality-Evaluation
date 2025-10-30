def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
        
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    # Create a mapping from value to its index for quick lookup.
    pos = [0] * (n + 1)
    for i in range(n):
        pos[A[i]] = i
        
    operations = []
    
    # Iterate through each position and swap if the element doesn't match its correct value.
    for i in range(n):
        if A[i] != i + 1:
            j = pos[i + 1]  # Find the current position of the correct value.
            operations.append((i + 1, j + 1))  # Record the operation (1-indexed).
            
            # Swap elements at i and j.
            ai = A[i]
            A[i], A[j] = A[j], A[i]
            
            # Update the positions of the swapped values.
            pos[ai] = j
            pos[i + 1] = i
            
    # Prepare the output.
    output = [str(len(operations))]
    for op in operations:
        output.append(f"{op[0]} {op[1]}")
    sys.stdout.write("
".join(output))
    
if __name__ == '__main__':
    main()