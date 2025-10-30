import sys

def main():
    # Read N, the number of elements.
    N = int(sys.stdin.readline())
    
    # Read the sequence A as a list of integers.
    A = [int(x) for x in sys.stdin.readline().split()]

    # Initialize min_len to a value larger than any possible valid length.
    # float('inf') is a common choice. N+1 would also work if N >= 1.
    min_len = float('inf')
    
    # Dictionary to store the last seen index of each value.
    # Format: {value: index}
    last_occurrence = {}

    # Iterate through the array with index i and value val=A[i].
    for i in range(N):
        val = A[i]
        
        # Check if this value has been seen before.
        if val in last_occurrence:
            # If yes, a subarray with repeated 'val' is found.
            # It starts at the previous occurrence: last_occurrence[val]
            # It ends at the current occurrence: i
            # The length of this subarray A[last_occurrence[val]...i] is i - last_occurrence[val] + 1.
            length = i - last_occurrence[val] + 1
            
            # Update min_len if this newly found subarray is shorter.
            if length < min_len:
                min_len = length
        
        # Update the last seen index of 'val' to the current index 'i'.
        # This is crucial: by always using the most recent previous occurrence (which is what
        # last_occurrence[val] stores before this update), we ensure that the calculated
        # length (i - last_occurrence[val] + 1) corresponds to a subarray bounded by
        # two consecutive occurrences of 'val', which is what we need for the shortest length.
        last_occurrence[val] = i

    # After checking all elements, if min_len is still float('inf'),
    # it means no repeated value was found in any subarray.
    if min_len == float('inf'):
        print(-1)
    else:
        # Otherwise, print the minimum length found.
        print(min_len)

# Standard boilerplate to run the main function.
if __name__ == '__main__':
    main()