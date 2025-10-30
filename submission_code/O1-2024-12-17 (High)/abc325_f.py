def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    D = list(map(int, data[1:N+1]))
    idx = N + 1
    
    L1 = int(data[idx]); C1 = int(data[idx+1]); K1 = int(data[idx+2])
    L2 = int(data[idx+3]); C2 = int(data[idx+4]); K2 = int(data[idx+5])
    
    # A helper function to compute how many type-2 sensors 
    # are needed if k type-1 sensors are allocated to cover
    # a section of length d.
    #
    # leftover = d - k*L1
    # if leftover <= 0, then 0 type-2 sensors needed
    # else leftover is covered by type-2 sensors of length L2 each
    # needed2 = ceil(leftover / L2)
    def needed2(d, k):
        leftover = d - k * L1
        if leftover <= 0:
            return 0
        return (leftover + L2 - 1) // L2

    # x_i[i] = how many type-1 sensors currently allocated to section i
    x_i = [0]*N

    # curr_needed2[i] = how many type-2 sensors needed for section i 
    # given the current x_i[i] type-1 sensors allocated
    curr_needed2 = [needed2(D[i], 0) for i in range(N)]
    
    # sum2 = total type-2 sensors currently needed across all sections
    sum2 = sum(curr_needed2)

    # dp[x] will store the minimum total type-2 usage if we
    # allocate exactly x type-1 sensors (in the best possible way).
    dp = [0]*(K1 + 1)
    dp[0] = sum2

    # We use a max-heap (in Python, we'll store negative values to simulate max-heap).
    # Each entry is (-marginal_benefit, section_index).
    #
    # marginal_benefit = how many type-2 sensors we save by adding
    # one more type-1 sensor to a particular section i:
    #     curr_needed2[i] - needed2(d, x_i[i] + 1)
    import heapq
    heap = []
    
    # Initialize the heap with the current marginal benefits
    for i in range(N):
        next_val = needed2(D[i], 1)
        mb = curr_needed2[i] - next_val
        heapq.heappush(heap, (-mb, i))

    # Distribute the x-th type-1 sensor (for x from 1..K1) one by one
    for x in range(1, K1 + 1):
        # Pop until we find a top-of-heap that matches the current marginal benefit
        while True:
            neg_mb, i = heapq.heappop(heap)
            mb = -neg_mb
            # Recompute the actual marginal benefit in case the heap entry was outdated
            actual_mb = curr_needed2[i] - needed2(D[i], x_i[i] + 1)
            if mb == actual_mb:
                # Found the correct (non-outdated) entry
                break
            # Otherwise, keep popping

        # Allocate one more type-1 sensor to that section
        sum2 -= mb  # we save mb type-2 sensors
        x_i[i] += 1
        # Update that section's current needed2
        curr_needed2[i] -= mb

        # Now compute its new marginal benefit (for possibly adding yet another sensor)
        next_val = needed2(D[i], x_i[i] + 1)
        new_mb = curr_needed2[i] - next_val
        heapq.heappush(heap, (-new_mb, i))

        # Store the resulting total type-2 usage in dp[x]
        dp[x] = sum2

    # Finally, find the minimum total cost among all feasible x
    # where the required number of type-2 sensors (dp[x]) is <= K2
    ans = None
    for x in range(K1 + 1):
        if dp[x] <= K2:
            cost = x * C1 + dp[x] * C2
            if ans is None or cost < ans:
                ans = cost

    # Print the result
    print(-1 if ans is None else ans)

# Don't forget to call main()
if __name__ == "__main__":
    main()