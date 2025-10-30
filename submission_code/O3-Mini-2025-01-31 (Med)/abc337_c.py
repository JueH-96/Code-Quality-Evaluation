def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))
    
    # Create an array to store the person who stands right behind each person. 
    # Indexing from 1 to n; a value of 0 indicates no one stands behind that person.
    behind = [0] * (n + 1)
    
    front = None  # This will hold the person at the front of the line.
    
    for i in range(1, n + 1):
        info = A[i - 1]
        if info == -1:
            front = i
        else:
            behind[info] = i
    
    # Build the line starting from the person at the front
    result = []
    current = front
    while current:
        result.append(current)
        current = behind[current]  # Move to the person right behind the current one
    
    # Print the resulting order separated by spaces
    print(" ".join(map(str, result)))
    
if __name__ == '__main__':
    main()