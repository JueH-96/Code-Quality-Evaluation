# YOUR CODE HERE
def find_sequence_b(n, a):
    # Dictionary to store the last occurrence of each element
    last_occurrence = {}
    # List to store the result sequence B
    b = []

    # Iterate over each element in the sequence A
    for i in range(n):
        # Get the current element
        current_element = a[i]
        
        # Check if the current element has appeared before
        if current_element in last_occurrence:
            # If yes, append the last occurrence index to B
            b.append(last_occurrence[current_element])
        else:
            # If no, append -1 to B
            b.append(-1)
        
        # Update the last occurrence of the current element
        last_occurrence[current_element] = i + 1  # Store 1-based index

    return b

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    # First line is the number of elements N
    n = int(data[0])
    # The rest are the elements of the sequence A
    a = list(map(int, data[1:n+1]))
    
    # Find the sequence B
    result = find_sequence_b(n, a)
    
    # Print the result
    print(" ".join(map(str, result)))