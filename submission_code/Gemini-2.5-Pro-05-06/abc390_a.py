# YOUR CODE HERE
# Read input
A = list(map(int, input().split()))

# Target sorted sequence
S = [1, 2, 3, 4, 5]

# Default answer
final_answer = "No"

if A == S:
    # Already sorted, requires 0 operations. Problem asks for exactly 1.
    # final_answer is already "No", so no change needed here.
    # This explicit assignment is for clarity of the condition.
    final_answer = "No" 
else:
    # Not sorted. Try to sort with one adjacent swap.
    # Iterate through all possible adjacent swap positions.
    # For a list of length 5, indices are 0,1,2,3,4.
    # Pairs to swap involve indices:
    # (0,1), (1,2), (2,3), (3,4)
    # The loop for i should go from 0 to len(A)-2 (which is 3 for len=5).
    # range(len(A) - 1) correctly gives i = 0, 1, 2, 3.
    for i in range(len(A) - 1):
        # Create a mutable copy of A for this iteration's swap
        A_candidate = list(A)
        
        # Perform the adjacent swap
        A_candidate[i], A_candidate[i+1] = A_candidate[i+1], A_candidate[i]
        
        # Check if this swapped version is the sorted sequence S
        if A_candidate == S:
            final_answer = "Yes"
            break  # Found a solution, no need to check other swaps

# Print the final determined answer
print(final_answer)