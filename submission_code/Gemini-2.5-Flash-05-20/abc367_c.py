import sys

def solve():
    # Read N and K from the first line
    N, K = map(int, sys.stdin.readline().split())
    # Read the R_i values from the second line
    R = list(map(int, sys.stdin.readline().split()))

    # This list will store the elements of the current sequence being built
    # We don't need a separate list for results as we print directly
    # when a valid sequence is found, ensuring lexicographical order.
    current_sequence = []

    # Recursive function to generate sequences
    # index: current position (0-indexed) in the sequence to fill
    # current_sum: sum of elements in current_sequence so far
    def generate_sequences(index, current_sum):
        # Base case: A sequence of length N has been formed
        if index == N:
            # Check if the sum of elements is a multiple of K
            if current_sum % K == 0:
                # Print the sequence. Using map(str, ...) to convert integers to strings
                # and " ".join(...) to format them with spaces.
                # sys.stdout.write is generally faster for competitive programming than print().
                sys.stdout.write(" ".join(map(str, current_sequence)) + "
")
            return

        # Recursive step: Iterate through all possible values for the current element
        # R[index] gives the maximum value for the element at 'index'
        for val in range(1, R[index] + 1):
            # Add the current value to the sequence
            current_sequence.append(val)
            # Recurse for the next position, updating the sum
            generate_sequences(index + 1, current_sum + val)
            # Backtrack: Remove the last added value to try the next possibility
            # This is crucial for exploring all combinations
            current_sequence.pop()

    # Initial call to start generating sequences from the first position (index 0)
    # with an empty sequence and a sum of 0.
    generate_sequences(0, 0)

# Call the solve function to run the program
solve()