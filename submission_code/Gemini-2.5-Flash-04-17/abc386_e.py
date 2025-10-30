# YOUR CODE HERE
import itertools
import sys

# Function to calculate XOR sum of elements at given indices
def calculate_xor_sum_at_indices(A, indices):
    """Calculates the XOR sum of elements in A at the specified indices."""
    xor_sum = 0
    for i in indices:
        xor_sum ^= A[i]
    return xor_sum

def main():
    # Read N and K
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    K = int(line1[1])

    # Read the array A
    A = list(map(int, sys.stdin.readline().split()))

    max_xor_sum = 0

    # We choose the smaller set to iterate over combinations.
    # If K <= N/2, iterate over combinations of K elements to choose.
    # If K > N/2, iterate over combinations of N-K elements (to exclude).
    # Using integer division N // 2 correctly determines which is smaller.
    if K <= N // 2:
        k_to_iterate = K
        # The number of combinations is C(N, K) <= 10^6
        # The cost per combination is O(K)
        for indices in itertools.combinations(range(N), k_to_iterate):
            current_xor_sum = calculate_xor_sum_at_indices(A, indices)
            max_xor_sum = max(max_xor_sum, current_xor_sum)
    else:
        # Iterate through combinations of N-K elements to exclude
        k_to_iterate = N - K
        # The number of combinations is C(N, N-K) = C(N, K) <= 10^6
        # The cost per combination is O(N-K)

        # Calculate the total XOR sum of all elements
        total_xor_sum = 0
        for x in A:
            total_xor_sum ^= x

        # If k_to_iterate is 0 (i.e., K=N), the only combination is the empty set (0 excluded elements).
        # The XOR sum of chosen elements is the total XOR sum.
        if k_to_iterate == 0:
             max_xor_sum = total_xor_sum
        else:
            # Iterate through combinations of indices to exclude
            # For each combination of excluded elements, calculate their XOR sum.
            # The XOR sum of the chosen elements is total_xor_sum ^ xor_sum_excluded.
            # We want to maximize this value.
            # Initialize max_xor_sum to 0. Since A_i >= 0, XOR sum >= 0.
            max_xor_sum = 0

            for indices_excluded in itertools.combinations(range(N), k_to_iterate):
                xor_sum_excluded = calculate_xor_sum_at_indices(A, indices_excluded)
                current_xor_sum_chosen = total_xor_sum ^ xor_sum_excluded
                max_xor_sum = max(max_xor_sum, current_xor_sum_chosen)

    # Print the result
    print(max_xor_sum)

# Execute the main function
main()