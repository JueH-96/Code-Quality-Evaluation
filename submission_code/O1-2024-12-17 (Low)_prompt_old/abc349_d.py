def solve():
    import sys
    data = sys.stdin.read().strip().split()
    L, R = map(int, data)

    # We'll greedily take, from the current left endpoint 'curr',
    # the largest "power-of-two-aligned" interval [curr, curr+size)
    # that fits inside [L, R). A sequence is "good" if it corresponds
    # to S(2^i * j, 2^i * (j+1)), i.e., length = 2^i and start is multiple of 2^i.

    # Function to get the largest power of two <= x.
    # For x > 0, this returns 2^floor(log2(x)).
    def largest_power_of_two_leq(x):
        # One way: 1 << (x.bit_length() - 1)
        return 1 << (x.bit_length() - 1)

    intervals = []
    curr = L

    while curr < R:
        # If curr == 0, everything divides 0, so we pick the largest power of two <= (R - curr).
        if curr == 0:
            size = largest_power_of_two_leq(R - curr)
        else:
            # size is the largest power of two dividing curr (lowest set bit of curr).
            # in Python, (curr & -curr) yields that power of two, but watch if it exceeds (R - curr).
            size = curr & -curr

            # If for some reason curr & -curr == 0 (shouldn't happen unless curr==0), handle that:
            if size == 0:
                size = largest_power_of_two_leq(R - curr)

        # Make sure size does not exceed what's left (R - curr).
        while size > R - curr:
            size >>= 1

        intervals.append((curr, curr + size))
        curr += size

    print(len(intervals))
    for l_i, r_i in intervals:
        print(l_i, r_i)

def main():
    solve()

if __name__ == "__main__":
    main()