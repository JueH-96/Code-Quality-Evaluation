def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:2+N+M]))

    # Sort the requirements in ascending order (smallest to largest)
    B.sort()

    # We'll use a Fenwick (Binary Indexed) Tree approach with coordinate compression
    # because A_i and B_i can be as large as 1e9.

    # 1) Coordinate-compress the box capacities A
    #    We'll store them in ascending order and keep counts of each distinct capacity.
    uniqueA = sorted(set(A))
    # Map from capacity -> index in Fenwicks
    # zero-based index
    index_of = {}
    for i, val in enumerate(uniqueA):
        index_of[val] = i

    # Fenwicks tree utility functions (1-based indexing internally)
    def fenwicks_build(size):
        return [0] * (size+1)

    def fenwicks_update(fw, i, val):
        while i < len(fw):
            fw[i] += val
            i += (i & -i)

    def fenwicks_sum(fw, i):
        s = 0
        while i > 0:
            s += fw[i]
            i -= (i & -i)
        return s

    # Build Fenwicks with frequency of each capacity
    fenwicks = fenwicks_build(len(uniqueA))
    # Populate Fenwicks counts
    for cap in A:
        pos = index_of[cap] + 1  # 1-based
        fenwicks_update(fenwicks, pos, 1)

    # Function to find the smallest Fenwicks index >= "left+1"
    # such that fenwicks_sum(mid) > leftSum.
    # Returns that index or -1 if none found.
    def fenwicks_find_first(fw, left, leftSum):
        lo = left + 1
        hi = len(fw) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if fenwicks_sum(fw, mid) > leftSum:
                hi = mid
            else:
                lo = mid + 1
        if fenwicks_sum(fw, lo) > leftSum:
            return lo
        return -1

    # Now try to assign each requirement B[i] with the smallest capacity >= B[i].
    import bisect
    total_cost = 0

    for b in B:
        # Find the leftmost capacity index that can handle b
        pos = bisect.bisect_left(uniqueA, b)
        if pos == len(uniqueA):
            # No capacity >= b
            print(-1)
            return
        # We want the smallest index i >= pos that has a positive count in Fenwicks.
        # leftSum = number of boxes used up to index pos-1
        # So if Fenwicks prefix sum at pos is S, we want the first i so that Fenwicks_sum(i) > S.
        leftSum = fenwicks_sum(fenwicks, pos)  # prefix sum up to pos (1-based means pos is exactly the partial we need)
        idx_1 = fenwicks_find_first(fenwicks, pos, leftSum)
        if idx_1 == -1:
            # no available box with capacity >= b
            print(-1)
            return

        # idx_1 is 1-based index, convert to 0-based
        idx_0 = idx_1 - 1
        # This capacity is uniqueA[idx_0]
        total_cost += uniqueA[idx_0]
        # Use this box up
        fenwicks_update(fenwicks, idx_1, -1)

    print(total_cost)


# Call main() to start the solution
if __name__ == "__main__":
    main()