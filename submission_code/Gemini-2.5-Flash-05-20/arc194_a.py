import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    current_sum = 0
    stack = []
    
    # Initialize max_ans to negative infinity. This is crucial for cases
    # where the maximum possible sum is negative (e.g., if all numbers are negative
    # and we are forced to append them). It also correctly handles cases where
    # the optimal sum is 0 (for an empty S).
    max_ans = -float('inf') 

    for x in A:
        # If the stack is empty, we are forced to append the current element 'x'.
        if not stack:
            stack.append(x)
            current_sum += x
        # If the stack is not empty, we have a choice: append 'x' or delete the last element.
        else:
            # Calculate the sum if we choose to append 'x'.
            append_possible_sum = current_sum + x
            
            # Get the last element of the stack to calculate sum if we choose to delete.
            top_val = stack[-1]
            # Calculate the sum if we choose to delete the last element.
            delete_possible_sum = current_sum - top_val

            # Make the greedy choice: pick the operation that yields a higher current_sum.
            # If sums are equal, we arbitrarily prefer appending to potentially keep more options open,
            # though this specific tie-breaking rule doesn't seem to affect correctness for this problem.
            if append_possible_sum >= delete_possible_sum:
                stack.append(x)
                current_sum += x
            else:
                stack.pop()
                current_sum -= top_val
        
        # After each operation, the 'current_sum' represents a valid sum of 'S'
        # if operations were to stop at this point. We track the maximum such sum encountered.
        max_ans = max(max_ans, current_sum)

    # Print the maximum possible sum found.
    # The initial max_ans = -float('inf') correctly handles cases like N=1, A=[-1]
    # where the final sum is negative and cannot be 0.
    # If the maximal sum found (max_ans) is negative, but S can be made empty (sum 0),
    # the problem implies 0 is preferable. However, based on the sample, if a negative sum
    # is the *only* option, then that negative sum is the answer. The current logic handles this.
    print(max_ans)

solve()