import sys

def solve():
    N = int(sys.stdin.readline())
    X = list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())

    # current_Y stores the transformed positions y_k = X_k - (k+1) (using 0-indexed k)
    # The road positions X_k are 1-indexed in problem description.
    # Person 1 is X_1. So for 0-indexed array current_X, current_X[k] means (k+1)-th person.
    # y_k = X_k - (k+1)
    current_Y = [X[k] - (k + 1) for k in range(N)]

    total_movement = 0

    for _ in range(Q):
        T, G = map(int, sys.stdin.readline().split())
        T_idx = T - 1  # Convert to 0-indexed person index

        # Calculate the new target y value for person T_idx
        # new_Y_T = G - (T_idx + 1)
        new_Y_T = G - T

        # Calculate the movement cost for person T_idx itself
        cost_for_query = abs(new_Y_T - current_Y[T_idx])
        
        # Update y_T_idx in the array
        current_Y[T_idx] = new_Y_T

        # Apply the PAVA-like rule:
        # For elements to the left of T_idx (from T_idx-1 down to 0):
        # Y'_k = min(Y_k, Y'_{k+1})
        # This means elements are "pulled" to the right if they are too far left
        # to maintain the non-decreasing order with respect to the element to their right.
        
        # We iterate from right to left, maintaining a "bound" from the right.
        # This bound ensures `Y'_k <= Y'_{k+1}`.
        # The `current_y_bound` represents `Y'_{k+1}` from the previous iteration.
        current_y_bound = current_Y[T_idx]
        for k in range(T_idx - 1, -1, -1):
            if current_Y[k] > current_y_bound:
                cost_for_query += (current_Y[k] - current_y_bound)
                current_Y[k] = current_y_bound
            current_y_bound = current_Y[k] # This ensures the next element to its left won't be larger than current_Y[k]

        # For elements to the right of T_idx (from T_idx+1 up to N-1):
        # Y'_k = max(Y_k, Y'_{k-1})
        # This means elements are "pulled" to the left if they are too far right
        # to maintain the non-decreasing order with respect to the element to their left.

        # We iterate from left to right, maintaining a "bound" from the left.
        # This bound ensures `Y'_k >= Y'_{k-1}`.
        # The `current_y_bound` represents `Y'_{k-1}` from the previous iteration.
        current_y_bound = current_Y[T_idx]
        for k in range(T_idx + 1, N):
            if current_Y[k] < current_y_bound:
                cost_for_query += (current_y_bound - current_Y[k])
                current_Y[k] = current_y_bound
            current_y_bound = current_Y[k] # This ensures the next element to its right won't be smaller than current_Y[k]
        
        total_movement += cost_for_query

    sys.stdout.write(str(total_movement) + '
')

solve()