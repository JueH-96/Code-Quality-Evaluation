# YOUR CODE HERE
import sys

def solve():
    # Read the number of mochi
    N = int(sys.stdin.readline())
    
    # Read the list of mochi sizes. The input guarantees they are sorted in ascending order.
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize the total count of possible kagamimochi pairs
    total_count = 0
    
    # Initialize a pointer 'i_ptr'. This pointer will keep track of the count of mochi
    # that satisfy the size condition for being a top mochi.
    # Specifically, after the inner `while` loop finishes for a given bottom mochi A[j],
    # `i_ptr` will be the count of indices k such that A[k] <= A[j] // 2.
    i_ptr = 0  
    
    # Iterate through each mochi A[j] using its index 'j'. We consider A[j] as the potential bottom mochi.
    # 'j' ranges from 0 to N-1.
    for j in range(N):
        # Calculate the maximum allowed size for a mochi to be placed on top of A[j].
        # The condition is size_top <= size_bottom / 2. We use integer division `//`.
        target = A[j] // 2 
        
        # This inner `while` loop advances the pointer `i_ptr`.
        # Since the array A is sorted and the `target` value (A[j] // 2) is non-decreasing as `j` increases,
        # `i_ptr` only moves forward across all iterations of the outer loop. This is the core
        # idea of the two-pointer approach, which makes the overall time complexity O(N).
        # We advance `i_ptr` as long as the element A[i_ptr] satisfies the size condition
        # (A[i_ptr] <= target) and `i_ptr` is still within the bounds of the array A.
        while i_ptr < N and A[i_ptr] <= target:
            i_ptr += 1
        
        # After the `while` loop finishes, `i_ptr` represents the number of elements
        # A[0], A[1], ..., A[i_ptr-1] that are less than or equal to `target`.
        # Each of these `i_ptr` elements can be placed on top of A[j].
        # The condition A[i] <= A[j] // 2 implies A[i] < A[j] (since A[i] >= 1).
        # Because A is sorted non-decreasingly, A[i] < A[j] implies i < j.
        # Thus, choosing any index k from 0 to i_ptr-1 automatically satisfies k < j,
        # ensuring that we are choosing two distinct mochi (indices k and j are different).
        # We add this count (`i_ptr`) to the `total_count`.
        total_count += i_ptr

    # Print the final total count of valid kagamimochi pairs.
    print(total_count)

# Execute the solve function to run the program.
solve()

# END YOUR CODE HERE