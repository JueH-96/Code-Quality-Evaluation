from itertools import accumulate, chain, repeat

def solvetrying(A):
    N = len(A)
    
    # Case 1: Non-decreasing sequence from the beginning
    min_lefts = tuple(chain((A[0],), (min_left + 1 for min_left in accumulate(A[1:], max)), repeat(10**18)))
    
    # Case 2: Non-decreasing sequence after a small value, with compensations
    max_lefts = list(accumulate(reversed(min_lefts), min))[::-1][:-1]
    max_right_compensates = [right_comp + 1 for right_comp in accumulate(chained_A := [A[0]] + list(A[1:] + [10**18]), min) if right_comp < right_max for right_comp, right_max in zip(chained_A, max_lefts)][::-1] + [10**18]
    
    try:
        current_right = max_compensate = 0
        for A_index, A_value in enumerate(A):
            # We haven't finished making the current value large enough
            if max_compensate < A_value:
                # Find the smallest max compensated value in the list where we can compensate with the current target value
                # We still need to account for the current target + the current value.
                max_compensate = min(max(max_compensate, A_value - current_right), max_right_compensates[current_right:])
                # Compensate the current value to the max compensated value
                A_value = max_compensate + current_right
            else:
                A_value = max_compensate
            # Update the current value after compensating for the target operation.
            # If this is the last item and it is greater than our target, we have failed.
            if A_index == len(A) - 1 and A_value > max_compensate:
                return False
            
            # Set the current item to the new value and then move on to the next one.
            A[A_index] = A_value
            current_right += 1
        return True
    except IndexError:
        return False

# This code provides an easier way to write the repeated logic of running our function
# on multiple test cases read from stdin and writing each result back to stdout.
if __name__ == '__main__':
    T = int(input().strip())
    for _ in range(T):
        # Read input
        N = int(input().strip())
        A = list(map(int, input().strip().split()))
        
        # Solve the problem
        result = solvetrying(A)
        
        # Write result
        print('Yes' if result else 'No')