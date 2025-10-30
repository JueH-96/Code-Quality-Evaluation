def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1].strip()
    
    # Get the indices (0-indexed) of all 1s in S.
    ones = [i for i, ch in enumerate(S) if ch == '1']
    
    # If the ones are already contiguous, then swaps required is 0.
    # (Although problem guarantees at least one 1, we handle the edge case)
    if not ones:
        sys.stdout.write("0")
        return

    # Choose the median index of the ones
    k = len(ones)
    m = k // 2  # median index in the ones list
    median = ones[m]
    
    # We want to group the ones in contiguous positions.
    # An optimal strategy is to bring the ones to positions
    # [target, target+1, ..., target+k-1] where we set the median one's
    # target position equal to its current position.
    # Thus, the expected target index for the i-th 1 (0-index in ones) is:
    #   expected[i] = median - (m - i)
    # The minimal number of swaps is then the sum of absolute differences:
    #   cost = sum(|ones[i] - expected[i]|) for all i.
    
    swaps = 0
    for i in range(k):
        target = median - (m - i)
        swaps += abs(ones[i] - target)
    
    sys.stdout.write(str(swaps))

if __name__ == '__main__':
    main()