def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # First integer is N
    N = int(input_data[0])
    
    # The next 3N integers represent the sequence A.
    A = list(map(int, input_data[1:]))

    # Dictionary to store the middle occurrence index (f(i)) for each number i.
    # We'll use a list for occurrence counts (indexed from 1 to N).
    occurrence_count = [0] * (N + 1)
    f_index = [0] * (N + 1)  # f_index[i] will store the index of the second occurrence of i

    # Iterate through the array to record the index when each number appears for the second time.
    for idx, num in enumerate(A, start=1):  # Using 1-indexed positions.
        occurrence_count[num] += 1
        if occurrence_count[num] == 2:
            f_index[num] = idx
    
    # Create the list of numbers 1 to N and sort them by their middle occurrence index.
    result = sorted(range(1, N + 1), key=lambda i: f_index[i])
    
    # Print the result separated by spaces.
    print(" ".join(map(str, result)))

# Call the main function
if __name__ == '__main__':
    main()