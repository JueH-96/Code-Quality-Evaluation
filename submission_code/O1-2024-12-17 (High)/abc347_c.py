def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = int(data[1])
    B = int(data[2])
    D = list(map(int, data[3:]))

    # The length of one "week"
    C = A + B

    # For each plan i, define offset_i = D_i mod C.
    # Then let s_i = (-offset_i) mod C.  (These s_i represent where Takahashi's
    # "today" could be so that (today + D_i) lands in a holiday.)
    s = []
    for d in D:
        offset = d % C
        s.append((-offset) % C)

    # We only need to cover each distinct value of s_i.  Duplicates don't add new constraints.
    s_unique = sorted(set(s))
    M = len(s_unique)

    # If there is only one distinct s_i, we can always schedule all on holidays,
    # because one arc of length A trivially covers that single point.
    if M == 1:
        print("Yes")
        return

    # We want to check if all distinct s_i can fit into some arc of length A on the circle of length C.
    #
    # A standard way: make an extended array of length 2*M, where we append C to each of the first M elements.
    # Then, for each i in [0..M-1], look at the block of M consecutive elements (i through i+M-1).
    # If the difference between the last and first in that block is < A, it means there is an
    # arc of length A covering all distinct points (hence all plans).
    s_extended = s_unique[:]
    for i in range(M):
        s_extended.append(s_unique[i] + C)

    # Check each subarray of length M
    # If we ever find that the difference is < A, that means we can place an arc of length A
    # that covers all distinct s_i, so answer "Yes".
    for i in range(M):
        # s_extended[i+M-1] - s_extended[i] is the distance covering M points
        if s_extended[i + M - 1] - s_extended[i] < A:
            print("Yes")
            return

    # If no suitable arc was found, answer "No".
    print("No")


# Do not forget to call main().
if __name__ == "__main__":
    main()