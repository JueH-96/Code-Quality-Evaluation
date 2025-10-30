# YOUR CODE HERE
import sys

def solve():
    """
    Solves the AtCoder Kingdom problem by analyzing the gaps between plan days on a circle.
    """
    try:
        # Read problem parameters from standard input
        # N: number of plans
        # A: number of holidays in a week
        # B: number of weekdays in a week
        line1 = sys.stdin.readline()
        if not line1: return
        N, A, B = map(int, line1.split())

        # Read the schedule of plans
        # D_i: the i-th plan is scheduled D_i days later
        line2 = sys.stdin.readline()
        if not line2: return
        D = list(map(int, line2.split()))
    except (IOError, ValueError):
        # Handle cases with empty or malformed input, common in some environments
        return

    # A week has C = A + B days.
    C = A + B

    # We determine the positions of all plans within a single week cycle.
    # Using a set handles duplicate remainders efficiently.
    # The remainders are then sorted to analyze the gaps between them.
    remainders = sorted(list(set(d % C for d in D)))
    
    m = len(remainders)

    # If all plans fall on the same day of the week (m <= 1), we can always
    # choose 'today' to make that day a holiday (e.g., day 0).
    if m <= 1:
        print("Yes")
        return
        
    # Calculate the gaps between consecutive scheduled days on the circle.
    gaps = []
    for i in range(m - 1):
        gaps.append(remainders[i+1] - remainders[i])
    
    # The last gap is the "wrap-around" from the last day back to the first.
    gaps.append(C + remainders[0] - remainders[m-1])
    
    # The condition for all plans to possibly be on holidays is that we can find
    # a continuous block of B weekdays that fits into one of the empty gaps
    # between scheduled plan days. A gap of length `g` contains `g-1` empty days.
    # To fit B weekdays, we need a gap `g` such that `g-1 >= B`, which means `g >= B+1`.
    # Therefore, it's possible if and only if the maximum gap is large enough.
    if max(gaps) >= B + 1:
        print("Yes")
    else:
        print("No")

solve()