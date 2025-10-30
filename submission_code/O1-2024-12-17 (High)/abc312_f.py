def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    data = list(map(int, input_data[2:]))

    zero_cans = []
    one_cans = []
    openers = []

    # Parse input
    idx = 0
    for _ in range(N):
        T = data[idx]; X = data[idx+1]
        idx += 2
        if T == 0:
            zero_cans.append(X)
        elif T == 1:
            one_cans.append(X)
        else:
            openers.append(X)

    # Sort descending
    zero_cans.sort(reverse=True)
    one_cans.sort(reverse=True)
    openers.sort(reverse=True)

    # Prefix sums for zero-cans (happiness), one-cans (happiness), openers (capacity)
    def prefix_sums(arr):
        ps = [0]*(len(arr)+1)
        for i,v in enumerate(arr, start=1):
            ps[i] = ps[i-1] + v
        return ps

    zeroPS = prefix_sums(zero_cans)  # zeroPS[i] = sum of top i zero-cans
    onePS  = prefix_sums(one_cans)   # onePS[i]  = sum of top i one-cans
    openPS = prefix_sums(openers)    # openPS[i] = sum of top i opener capacities

    # We'll iterate over how many 1-cans (R) we actually "open"
    # We need the minimal number of openers L such that openPS[L] >= R.
    # We'll do a two-pointer approach to avoid O(M logN).
    best = 0
    l = 0  # pointer for how many openers we are using
    n_openers = len(openers)
    for R in range(min(M, len(one_cans)) + 1):
        needed = R
        # Increase l until openPS[l] >= needed or l > n_openers
        while l <= n_openers and l < n_openers and openPS[l] < needed:
            l += 1

        # If we've run out of openers and still not enough capacity, break
        if l > n_openers or openPS[l] < needed:
            # no more capacity for bigger R either
            break

        # Now openPS[l] >= needed, so we can open R 1-cans using l openers
        # We must pick R + l items so far
        if R + l <= M:
            # The remaining picks can be from 0-cans (which always give happiness)
            leftover = M - (R + l)
            leftover_zero = min(leftover, len(zero_cans))
            cand = onePS[R] + zeroPS[leftover_zero]
            if cand > best:
                best = cand
        else:
            # R+l > M means we can't pick them all
            break

    print(best)

# Do not forget to call main()
if __name__ == "__main__":
    main()