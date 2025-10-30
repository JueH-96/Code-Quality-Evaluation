import sys

def solve():
    # Read N from standard input.
    N = int(sys.stdin.readline())

    # Read the sequence A from standard input.
    # A is given as space-separated integers on a single line.
    # Convert them to a list of integers.
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize a list of lists to store the 0-based indices of occurrences for each number.
    # `occurrences[k]` will be a list of indices where the number `k` appears in A.
    # Since numbers are from 1 to N, we create N+1 sublists and use index 0 as a placeholder
    # to directly map number `k` to `occurrences[k]`.
    # Each number `k` occurs exactly three times, so each `occurrences[k]` will eventually
    # contain exactly 3 indices.
    occurrences = [[] for _ in range(N + 1)]

    # Iterate through the input array A with its 0-based indices.
    # For each value `val` at index `idx`:
    # Add `idx` to the list of occurrences for `val`.
    # Since we iterate `A` from left to right, indices will be appended in increasing order.
    for idx, val in enumerate(A):
        occurrences[val].append(idx)

    # Initialize a list to store tuples of (f(i), i).
    # This structure is chosen so that when sorted, it naturally sorts primarily by f(i).
    f_values_and_numbers = []

    # Iterate from 1 to N (inclusive) to calculate f(i) for each number i.
    for i in range(1, N + 1):
        # `occurrences[i]` now contains three 0-based indices where `i` appears in A.
        # For example, if `i` appears at indices `alpha`, `beta`, `gamma` (0-based)
        # where `alpha < beta < gamma`, then `occurrences[i]` will be `[alpha, beta, gamma]`.
        # The middle occurrence is thus at `occurrences[i][1]`.
        
        # The problem defines f(i) as the 1-based index of the middle occurrence.
        # So, we take the 0-based index `occurrences[i][1]` and add 1 to it.
        f_i = occurrences[i][1] + 1
        
        # Append the tuple (f_i, i) to our list.
        f_values_and_numbers.append((f_i, i))

    # Sort the list of tuples.
    # Python's default sort for tuples sorts lexicographically:
    # it sorts by the first element, then by the second element (if first elements are equal), and so on.
    # This means it will sort primarily by `f_i` (the middle occurrence index).
    f_values_and_numbers.sort()

    # After sorting, we need to extract only the original numbers (`i`) in the sorted order.
    # We do this using a list comprehension, taking the second element of each tuple.
    sorted_numbers = [num for f_val, num in f_values_and_numbers]

    # Join the sorted numbers with spaces and add a newline character.
    # Then write the result to standard output.
    sys.stdout.write(" ".join(map(str, sorted_numbers)) + "
")

# Call the solve function to execute the program.
if __name__ == '__main__':
    solve()