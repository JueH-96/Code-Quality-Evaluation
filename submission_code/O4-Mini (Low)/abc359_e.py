def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    H = list(map(int, input().split()))
    # ans[i] = the operation count when A_i first becomes positive
    # We maintain two values as we go left to right:
    #   total = the total number of operations performed so far
    #   depth = the current height of the "front" at the previous index
    #
    # Invariant after computing ans[i-1]:
    #   total == ans[i-1]
    #   depth == A_{i-1} at that time, which is exactly 1
    #
    # To make A_i positive, we must send a new "unit" from A_0 all the way to A_i.
    # That unit can advance from position k-1 to k only if
    #    current A_{k-1} > H_k.
    # Initially, at time total, we have A_{i-1} = 1.  We need to raise A_{i-1}
    # until it exceeds H[i], so that the next new unit will cross into i.
    #
    # Each new global operation increases A_0 by 1.  That "pulse" then
    # ripples right until it gets blocked.  But once A_{i-1} > H[i],
    # every subsequent operation will advance one unit into A_i.
    #
    # Thus, to get the first unit into A_i, we need to perform
    #   max(0, H[i] - depth + 1)  extra operations beyond 'total',
    # which raise depth from its current value up to H[i]+1.
    # Then one more operation to actually send the unit into A_i.
    #
    # So the increment in total is  (H[i] - depth + 1) + 1  =  H[i] - depth + 2.
    #
    # After that operation, A_i == 1, so for the next step depth resets to 1
    # and total is the answer for position i.
    #
    # Initialize total=0, depth=0 (A_0 is depth, but we only care when i=1).
    #
    total = 0
    depth = 0
    ans = []
    for h in H:
        # need extra = h - depth + 2, but if h < depth then h-depth+2<=2:
        # it still works: we need at least 1 op to push the unit
        extra = h - depth + 2
        if extra < 1:
            extra = 1
        total += extra
        ans.append(total)
        # after that A_i == 1
        depth = 1
    print(" ".join(map(str, ans)))

if __name__ == "__main__":
    main()