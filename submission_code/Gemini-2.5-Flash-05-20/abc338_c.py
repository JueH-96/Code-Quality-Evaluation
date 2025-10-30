# YOUR CODE HERE
import math

def solve():
    N = int(input())
    Q = list(map(int, input().split()))
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    max_total_servings = 0

    # Calculate the maximum possible servings of dish A if only A is made.
    # This value will serve as the upper bound for iterating 'x'.
    # If A_i is 0, it doesn't constrain 'x'.
    # The problem guarantees that at least one A_i >= 1, so max_servings_A_only will be finite.
    max_servings_A_only = float('inf')
    for i in range(N):
        if A[i] > 0:
            max_servings_A_only = min(max_servings_A_only, Q[i] // A[i])
    
    # Iterate through all possible integer servings of dish A (let's call it 'x').
    # 'x' can range from 0 up to the maximum possible servings of A.
    # The `+ 1` is because range() is exclusive on the upper bound.
    for x in range(int(max_servings_A_only) + 1):
        # Calculate the remaining ingredients after making 'x' servings of dish A.
        # Initialize an array for remaining ingredients.
        Q_remaining = [0] * N
        
        # Q_remaining[i] will always be non-negative because x is bounded by max_servings_A_only.
        # If A[i] is 0, then x * A[i] is 0, so Q_remaining[i] = Q[i] (which is >= 0).
        for i in range(N):
            Q_remaining[i] = Q[i] - x * A[i]

        # Calculate the maximum possible servings of dish B (let's call it 'y')
        # using the 'Q_remaining' ingredients.
        # If B_i is 0, it doesn't constrain 'y'.
        # The problem guarantees that at least one B_i >= 1, so max_possible_y_for_this_x will be finite.
        max_possible_y_for_this_x = float('inf')
        for i in range(N):
            if B[i] > 0:
                max_possible_y_for_this_x = min(max_possible_y_for_this_x, Q_remaining[i] // B[i])
        
        # 'y' is the maximum integer number of servings of dish B we can make.
        y = int(max_possible_y_for_this_x)

        # The current total number of servings for this combination of x and y.
        current_total_servings = x + y
        
        # Update the overall maximum total servings found so far.
        max_total_servings = max(max_total_servings, current_total_servings)

    # Print the final maximum total servings.
    print(max_total_servings)

# Call the solve function to run the program.
if __name__ == '__main__':
    solve()