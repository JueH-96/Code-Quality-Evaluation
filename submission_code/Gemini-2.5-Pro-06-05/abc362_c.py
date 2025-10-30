import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Read N, the number of pairs.
        # Constraints state 1 <= N, so we don't need to handle N=0.
        n_str = sys.stdin.readline()
        if not n_str: return
        n = int(n_str)
    except (ValueError, IndexError):
        # This handles cases like empty input.
        return

    # Read all N pairs of (L, R) and store them.
    # At the same time, calculate the sum of all lower bounds (sum_L)
    # and all upper bounds (sum_R).
    pairs = [list(map(int, sys.stdin.readline().split())) for _ in range(n)]
    
    sum_L = sum(p[0] for p in pairs)
    sum_R = sum(p[1] for p in pairs)

    # A solution exists if and only if the target sum (0)
    # lies within the possible range of sums [sum_L, sum_R].
    if not (sum_L <= 0 <= sum_R):
        print("No")
        return

    # If a solution is possible, print "Yes".
    print("Yes")

    # Construct one such solution using a greedy approach.
    # Start with the minimal sequence: X_i = L_i for all i.
    X = [p[0] for p in pairs]
    
    # The sum of this initial sequence is sum_L. We need to increase it to 0.
    # The total amount to add is -sum_L.
    amount_to_add = -sum_L

    # Distribute this required addition among the X_i's.
    for i in range(n):
        if amount_to_add == 0:
            break

        # The maximum possible increase for X[i] is R_i - L_i.
        possible_increase = pairs[i][1] - pairs[i][0]

        # The actual increase for X[i] is the minimum of what's
        # needed for the total sum and what's possible for this element.
        actual_increase = min(amount_to_add, possible_increase)

        # Apply the increase and update the remaining amount to add.
        X[i] += actual_increase
        amount_to_add -= actual_increase
    
    # Print the resulting sequence.
    print(*X)

if __name__ == "__main__":
    solve()