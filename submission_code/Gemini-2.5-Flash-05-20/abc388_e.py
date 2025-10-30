import sys

def solve():
    # Read N from standard input
    N = int(sys.stdin.readline())
    # Read the mochi sizes into a list A.
    # The problem states A is already sorted in ascending order.
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize a counter for the number of kagamimochi formed.
    count = 0

    # Initialize two pointers for the greedy approach:
    # 'i' iterates through potential A_top mochi (smaller sizes).
    # It starts at the beginning of the array.
    i = 0
    
    # 'j' iterates through potential A_bottom mochi (larger sizes).
    # It starts from the middle of the array, specifically at index N // 2.
    # This divides the array into two conceptual halves: A[0...N//2-1] and A[N//2...N-1].
    # We attempt to match an element from the first half (using pointer 'i')
    # with an element from the second half (using pointer 'j').
    j = N // 2

    # The loop continues as long as we have available mochi in both conceptual halves
    # to potentially form new pairs.
    # 'i < N // 2' ensures we don't go out of bounds for the first half and
    # that we consider at most N//2 'top' mochi.
    # 'j < N' ensures we don't go out of bounds for the array.
    while i < N // 2 and j < N:
        # Check the condition for forming a kagamimochi:
        # A mochi of size A[i] can be placed on top of A[j] if 2 * A[i] <= A[j].
        if 2 * A[i] <= A[j]:
            # If the condition is met, we successfully form a kagamimochi.
            # Increment the count.
            count += 1
            # Move both pointers forward, as both A[i] and A[j] are now used in a pair.
            # We look for the next potential A_top and A_bottom.
            i += 1
            j += 1
        else:
            # If 2 * A[i] > A[j], A[j] is too small to be the bottom mochi for A[i].
            # Since the array A is sorted, any mochi before A[j] (i.e., A[k] where k < j)
            # would also be too small (or even smaller) for A[i].
            # Thus, A[j] cannot be paired with the current A[i].
            # We must find a larger mochi for A[i]. So, we advance 'j' to the next
            # available larger mochi. 'i' remains in place, waiting for a suitable A_bottom.
            j += 1
    
    # After the loop, 'count' holds the maximum number of kagamimochi that can be formed.
    print(count)

# Call the solve function to run the program logic.
solve()