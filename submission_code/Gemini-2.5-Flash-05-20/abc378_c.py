import sys

def solve():
    # Read N from standard input
    N = int(sys.stdin.readline())
    
    # Read the sequence A from standard input
    # Use map(int, ...) to convert string parts to integers
    A = list(map(int, sys.stdin.readline().split()))

    # last_occurrence will store the 1-based index of the most recent occurrence
    # for each number encountered so far.
    # Key: value from A
    # Value: 1-based index of its last occurrence
    last_occurrence = {}
    
    # B will store the results as per the problem definition
    B = []

    # Iterate through the input sequence A using 0-based index `i`
    for i in range(N):
        current_value = A[i]
        
        # Check if the current_value has been seen before
        if current_value in last_occurrence:
            # If yes, its most recent previous position is stored in last_occurrence
            B.append(last_occurrence[current_value])
        else:
            # If no, this is its first occurrence, so append -1
            B.append(-1)
        
        # Update the last seen position for current_value to the current 1-based index (i + 1)
        last_occurrence[current_value] = i + 1
    
    # Print the elements of B, separated by spaces.
    # The * operator unpacks the list B into separate arguments for print().
    # The default separator for print() is a space.
    print(*(B))

# Call the solve function to execute the program
solve()