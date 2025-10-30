def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Faster iterator for large input
    it = 0
    def read_int():
        nonlocal it
        val = int(input_data[it])
        it += 1
        return val

    # Read N, K, Q
    N = read_int()
    K = read_int()
    Q = read_int()

    # Read all updates
    X = [0]*Q
    Y = [0]*Q
    vals = set()
    vals.add(0)  # include zero since A starts with all zeros
    for i in range(Q):
        x_i = read_int()
        y_i = read_int()
        X[i] = x_i - 1  # convert to 0-based index
        Y[i] = y_i
        vals.add(y_i)

    # Coordinate compression
    # Sort unique values
    coords = sorted(vals)          # 0-based array of unique sorted values
    # Create a dict mapping value -> compressed index (1-based for Fenwicks)
    valToIdx = {}
    for i, v in enumerate(coords):
        valToIdx[v] = i + 1
    M = len(coords)  # number of distinct values

    # Fenwicks: one for frequencies, one for sum of values
    fenwFreq = [0]*(M+1)  # 1-based
    fenwSum  = [0]*(M+1)  # 1-based

    def fenwicksum(bit, idx):
        s = 0
        while idx > 0:
            s += bit[idx]
            idx -= idx & -idx
        return s

    def fenwickupdate(bit, idx, delta):
        while idx <= M:
            bit[idx] += delta
            idx += idx & -idx

    # Precompute the largest power of two <= M for Fenwicks search
    maxPow = 1
    while maxPow <= M:
        maxPow <<= 1
    maxPow >>= 1

    # fenwicksSearch(bit, target):
    #   returns the largest index i in [0..M] s.t. fenwicksum(bit,i) < target
    #   if fenwicksum(bit,M) < target, we'll end up returning M
    def fenwicksSearch(bit, target):
        pos = 0
        step = maxPow
        while step > 0:
            nxt = pos + step
            if nxt <= M and bit[nxt] < target:
                target -= bit[nxt]
                pos = nxt
            step >>= 1
        return pos

    # Get the sum of the largest K elements
    # Using the Fenwicks, total number of elements is fenwicksum(fenwFreq,M).
    # If total <= K, we use sum of all. Otherwise we find the boundary so that
    # exactly K are counted from the top.
    def getTopKSum(k):
        total = fenwicksum(fenwFreq, M)
        # If K >= total, sum of top K is sum of all elements
        if k >= total:
            return fenwicksum(fenwSum, M)
        # We want to skip 'boundary = total - k' smallest elements
        boundary = total - k
        # Find largest j so fenwicksum(freq,j) < boundary
        j = fenwicksSearch(fenwFreq, boundary)
        # partialNeeded is how many items we still need to skip in the next bucket
        skipSoFar = fenwicksum(fenwFreq, j)
        partialNeeded = boundary - skipSoFar
        sumBelow = fenwicksum(fenwSum, j)
        sumAll = fenwicksum(fenwSum, M)

        # The next coordinate is j+1 in Fenwicks, but in 0-based for coords it's (j).
        valOfJp1 = coords[j]  # safe because j < M if boundary <= total

        # The sum of the boundary (smallest total-k) elements = sumBelow + (partialNeeded * valOfJp1)
        # So top k sum = sumAll - that
        return sumAll - (sumBelow + partialNeeded * valOfJp1)

    # Initialize Fenwicks with all A[i] = 0
    # freq of '0' is N; sum of '0' is 0
    idx0 = valToIdx[0]
    fenwickupdate(fenwFreq, idx0, N)  # N zeros
    # fenwSum doesn't change because 0*N = 0

    # Keep track of current values per index
    A = [0]*N

    out = []
    for i in range(Q):
        old_val = A[X[i]]
        new_val = Y[i]
        if old_val != new_val:
            # Remove old_val
            old_idx = valToIdx[old_val]
            fenwickupdate(fenwFreq, old_idx, -1)
            fenwickupdate(fenwSum, old_idx, -old_val)
            # Add new_val
            new_idx = valToIdx[new_val]
            fenwickupdate(fenwFreq, new_idx, 1)
            fenwickupdate(fenwSum, new_idx, new_val)
            A[X[i]] = new_val

        # Compute f(A) = sum of top K
        ans = getTopKSum(K)
        out.append(str(ans))

    print("
".join(out))

# Do not forget to call main()
main()