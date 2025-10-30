def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    t = int(input_data[0])
    index = 1
    results = []
    
    for _ in range(t):
        n = int(input_data[index])
        index += 1
        a = list(map(int, input_data[index:index+n]))
        index += n
        
        # We need to find the maximum sum of a non-empty subarray with alternating parities.
        best = -10**9  # since minimum values are at least -1000, so safe.
        # We'll use dp, where dp[i] is the max sum of alternating subarray ending at i.
        # However, we can do in O(n) space by tracking the last dp value and its parity.
        # But note that dp[i] depends on extension from dp[i-1] only if parity constraint holds.
        # We can maintain just previous dp and its ending element's parity.
        
        prev_sum = None
        prev_parity = None
        
        for num in a:
            # candidate to start a new subarray is always num itself.
            start_new = num
            if prev_sum is not None and (abs(prev_parity - (num & 1)) == 1):
                # If previous dp ended with an element of different parity,
                # then we can extend it
                extend = prev_sum + num
                curr_sum = max(start_new, extend)
            else:
                curr_sum = start_new
            # Update best
            if curr_sum > best:
                best = curr_sum
            # Update dp tracking: current dp value and its parity
            prev_sum = curr_sum
            prev_parity = num & 1  # parity: 0 for even, 1 for odd
        results.append(str(best))
    
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()