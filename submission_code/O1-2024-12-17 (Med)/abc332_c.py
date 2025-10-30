def main():
    import sys

    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    S = data[2]

    # We'll split the schedule into blocks separated by '0'.
    # Within each block (no washing days), we must use distinct T-shirts:
    # - "1" (meal) can be plain or logo
    # - "2" (CP event) must be logo
    #
    # For a block with Mcount = number of '1's and Lcount = number of '2's:
    #  1) At least Lcount distinct logo T-shirts are needed for the CP event days.
    #  2) If we have M plain T-shirts, but M < Mcount, we must use extra logo T-shirts
    #     to cover the leftover meal days: (Mcount - M).
    # So the requirement for a block is: Lcount + max(0, Mcount - M).
    # We want the maximum among all blocks as the final answer.

    max_needed = 0
    Mcount = 0
    Lcount = 0

    for i in range(N):
        if S[i] == '0':
            # End of a block
            if Mcount + Lcount > 0:
                needed = Lcount + max(0, Mcount - M)
                max_needed = max(max_needed, needed)
            # Start a new block
            Mcount = 0
            Lcount = 0
        else:
            if S[i] == '1':
                Mcount += 1
            else:  # S[i] == '2'
                Lcount += 1

    # Handle the final block if the string did not end with '0'
    if Mcount + Lcount > 0:
        needed = Lcount + max(0, Mcount - M)
        max_needed = max(max_needed, needed)

    print(max_needed)

# Call main() at the end
main()