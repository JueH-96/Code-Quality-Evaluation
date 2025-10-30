# YOUR CODE HERE
import sys

def solve():
    """
    Solves the AtCoder Regular Contest Rating problem by simulating the rating changes
    over a series of contests.
    """
    
    # Define the rating ranges for which updates apply in each division.
    # Div. 1: 1600 to 2799, inclusive.
    # Div. 2: 1200 to 2399, inclusive.
    RATING_UPDATE_RANGES = {
        1: (1600, 2799),
        2: (1200, 2399)
    }

    # Read the number of contests (N) and the initial rating (R).
    try:
        line = sys.stdin.readline()
        if not line:
            return
        N, R = map(int, line.split())
    except (ValueError, IndexError):
        # Graceful exit for malformed input, though not expected by problem constraints.
        return

    # Process each of the N contests sequentially.
    for _ in range(N):
        # Read the division (D) and performance (A) for the current contest.
        try:
            line = sys.stdin.readline()
            if not line:
                break
            D, A = map(int, line.split())
        except (ValueError, IndexError):
            continue

        # Get the rating range for the current division.
        # The problem guarantees D is either 1 or 2, so a KeyError is not expected.
        min_rating, max_rating = RATING_UPDATE_RANGES[D]

        # Check if the current rating is within the update range.
        if min_rating <= R <= max_rating:
            # If so, update the rating by the performance value.
            R += A

    # Print the final rating after all contests are finished.
    print(R)

solve()