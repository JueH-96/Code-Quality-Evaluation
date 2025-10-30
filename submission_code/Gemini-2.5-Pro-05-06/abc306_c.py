import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # positions[k] will store a list of 1-based indices where number k appears.
    # k ranges from 1 to N. So list size N+1, index 0 unused.
    positions = [[] for _ in range(N + 1)]

    # Iterate through the input array A.
    # enumerate gives 0-based index `idx_0based` and value `val`.
    # The problem uses 1-based indexing for positions in A, so we store idx_0based + 1.
    for idx_0based, val in enumerate(A):
        positions[val].append(idx_0based + 1)

    # The f(k) values are indices in A.
    # Smallest possible f(k) is 2 (e.g., for A = (k, k, k, ...)).
    # Largest possible f(k) is 3N-1 (e.g., for A = (..., k, k, k)).
    # We can use an array to map f(k) values back to k.
    # num_by_f_val[f_value] = k means that number k has f(k) = f_value.
    # Size 3*N + 1 for num_by_f_val array, to cover indices up to 3*N.
    # Initialize with 0, as valid numbers k are 1 to N (0 is not a valid number ID).
    num_by_f_val = [0] * (3 * N + 1) 

    for k_num in range(1, N + 1): # k_num is the number from 1 to N
        # Each list in positions[k_num] is guaranteed to have 3 elements by problem spec.
        # These elements are already sorted as they were added in order of appearance.
        # f(k_num) is the middle index, which is positions[k_num][1] (using 0-based list indexing).
        middle_occurrence_idx = positions[k_num][1]
        num_by_f_val[middle_occurrence_idx] = k_num
        
    # Iterate through possible f(k) values in increasing order (from 1 up to 3*N).
    # If num_by_f_val[idx] is not 0, it means idx was an f(k) value for number k = num_by_f_val[idx].
    # Collect these k values in order of their f(k) values.
    result_sequence = []
    # Iterate up to 3*N. Smallest f(k) is 2, largest is 3N-1.
    for f_val_candidate in range(1, 3 * N + 1): 
        if num_by_f_val[f_val_candidate] != 0: # Check if this index was an f(k) for some k
            result_sequence.append(num_by_f_val[f_val_candidate])
            
    # Print the resulting sequence of numbers, space-separated.
    # The print function with * unpacks the list, and items are converted to strings
    # and printed with spaces by default.
    print(*result_sequence)

if __name__ == '__main__':
    main()