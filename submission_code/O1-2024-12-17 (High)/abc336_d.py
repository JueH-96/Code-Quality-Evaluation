def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Edge case: if N == 1, answer is always 1 (as long as A[0] >= 1, which is given)
    if N == 1:
        print(1)
        return

    # ----------------------------------------------------------------
    # We will compute two arrays, left_arm and right_arm.
    #
    # left_arm[i] = length L of the longest "1,2,3,...,L" pattern that
    #               ends exactly at index i.  Concretely, this means
    #               for offset = 0..L-1:
    #                  A[i - offset] >= (offset + 1).
    #               Equivalently, if left_arm[i-1] = L0 and A[i] >= L0+1,
    #               we can extend by 1; otherwise we reset to 1 if A[i]>=1, or 0 if A[i]<1.
    #
    # right_arm[i] = length R of the longest "1,2,3,...,R" pattern that
    #                starts exactly at index i (when read left to right).
    #                We compute it by traversing from right to left with
    #                a similar logic: if right_arm[i+1] = R0 and A[i]>= R0+1,
    #                we can extend by 1; otherwise reset appropriately.
    #
    # Once we have left_arm[i] and right_arm[i], the largest pyramid that
    # peaks at i has size = min(left_arm[i], right_arm[i]).  We take the
    # maximum of that over all i.
    #
    # This works because a pyramid of size k has left side of length k
    # (matching "1,2,3,...,k"), and right side of length k (matching "k,k-1,...,1"),
    # overlapping by one in the middle.  So the peak is at index i, and we
    # need at least k on the left_arm side and k on the right_arm side.
    #
    # ----------------------------------------------------------------

    left_arm = [0]*N
    # Compute left_arm in a left-to-right sweep
    if A[0] >= 1:
        left_arm[0] = 1
    for i in range(1, N):
        if A[i] >= left_arm[i-1] + 1: 
            # We can extend the 1..(left_arm[i-1]) sequence by 1
            left_arm[i] = left_arm[i-1] + 1
        elif A[i] >= 1:
            # Cannot extend, but can start a fresh "1"
            left_arm[i] = 1
        else:
            # A[i] < 1, can't even start
            left_arm[i] = 0

    right_arm = [0]*N
    # Compute right_arm in a right-to-left sweep
    if A[N-1] >= 1:
        right_arm[N-1] = 1
    for i in range(N-2, -1, -1):
        if A[i] >= right_arm[i+1] + 1:
            right_arm[i] = right_arm[i+1] + 1
        elif A[i] >= 1:
            right_arm[i] = 1
        else:
            right_arm[i] = 0

    # The answer is the maximum over i of min(left_arm[i], right_arm[i])
    ans = 0
    for i in range(N):
        candidate = min(left_arm[i], right_arm[i])
        if candidate > ans:
            ans = candidate

    print(ans)

# Don't forget to call main()
if __name__ == "__main__":
    main()