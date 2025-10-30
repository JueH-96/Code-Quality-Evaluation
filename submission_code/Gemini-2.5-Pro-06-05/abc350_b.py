import sys

def solve():
    """
    Solves the Takahashi's Teeth problem by tracking which teeth have their state flipped.
    """
    # Read N, the total number of holes, and Q, the number of treatments.
    try:
        line1 = sys.stdin.readline()
        if not line1: return
        n, q = map(int, line1.split())
    except (IOError, ValueError):
        return

    # If there are no treatments, the number of teeth remains N.
    if q == 0:
        print(n)
        return

    # Read the list of hole numbers that receive a treatment.
    try:
        treatments = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return

    # A hole initially has a tooth. Its state flips with each treatment.
    # - Even treatments: final state is 'present' (same as initial).
    # - Odd treatments: final state is 'absent' (flipped from initial).
    #
    # We use a set to track holes treated an odd number of times.
    absent_holes = set()

    for t in treatments:
        # This operation toggles the presence of 't' in the set.
        # If t is in the set, it's removed (total treatments for t becomes even).
        # If not, it's added (total treatments for t becomes odd).
        if t in absent_holes:
            absent_holes.remove(t)
        else:
            absent_holes.add(t)

    # The final number of teeth is the total number of holes (N) minus
    # the number of holes that ended up empty.
    final_teeth_count = n - len(absent_holes)

    # Print the result to standard output.
    print(final_teeth_count)

solve()