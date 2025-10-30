def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = int(data[1])
    B = int(data[2])
    D = list(map(int, data[3:3+N]))
    
    week_length = A + B
    
    # We need to find a starting day such that all (D_i + start) mod week_length < A
    # Let's find the possible range for start mod week_length
    
    # For each D_i, the condition is (start + D_i) % week_length < A
    # Which is equivalent to start % week_length < A - D_i % week_length or start % week_length >= week_length - (D_i % week_length)
    
    # To find a common start that satisfies all conditions, we need to find the intersection of all possible ranges
    
    # Initialize the possible range for start mod week_length
    low = 0
    high = week_length - 1
    
    for d in D:
        # Calculate the remainder of d modulo week_length
        rem = d % week_length
        # The condition is (start + rem) % week_length < A
        # Which is equivalent to start % week_length < (A - rem) % week_length
        # Or start % week_length >= (week_length - rem) % week_length
        
        # The valid range for start % week_length is [0, (A - rem - 1) % week_length] or [(week_length - rem) % week_length, week_length - 1]
        
        # So the invalid range is [(A - rem) % week_length, (week_length - rem - 1) % week_length]
        
        # We need to find the intersection of all valid ranges
        
        # The invalid range is [invalid_low, invalid_high]
        invalid_low = (A - rem) % week_length
        invalid_high = (week_length - rem - 1) % week_length
        
        # If invalid_low > invalid_high, the invalid range wraps around
        if invalid_low > invalid_high:
            # The invalid range is [invalid_low, week_length - 1] and [0, invalid_high]
            # So the valid range is [invalid_high + 1, invalid_low - 1]
            # We need to intersect this with the current [low, high]
            # So the new low is max(low, invalid_high + 1)
            # The new high is min(high, invalid_low - 1)
            new_low = max(low, (invalid_high + 1) % week_length)
            new_high = min(high, (invalid_low - 1) % week_length)
            if new_low > new_high:
                # No valid range
                print("No")
                return
            low = new_low
            high = new_high
        else:
            # The invalid range is [invalid_low, invalid_high]
            # So the valid range is [0, invalid_low - 1] and [invalid_high + 1, week_length - 1]
            # We need to intersect this with the current [low, high]
            # So the new low is max(low, 0)
            # The new high is min(high, invalid_low - 1)
            # Or the new low is max(low, invalid_high + 1)
            # The new high is min(high, week_length - 1)
            # So the valid ranges are [max(low, 0), min(high, invalid_low - 1)] and [max(low, invalid_high + 1), min(high, week_length - 1)]
            # We need to find if any of these ranges is non-empty
            range1_low = max(low, 0)
            range1_high = min(high, invalid_low - 1)
            range2_low = max(low, invalid_high + 1)
            range2_high = min(high, week_length - 1)
            if range1_low > range1_high and range2_low > range2_high:
                # No valid range
                print("No")
                return
            # Update low and high to the union of the valid ranges
            # Since we need to find a common start that satisfies all conditions, we need to find the intersection of all valid ranges
            # So we need to find the intersection of all possible ranges
            # So for each step, we need to find the intersection of the current [low, high] with the valid ranges
            # So we need to find the intersection of [low, high] with [range1_low, range1_high] and [range2_low, range2_high]
            # So the new low is max(low, range1_low, range2_low)
            # The new high is min(high, range1_high, range2_high)
            new_low = max(low, range1_low, range2_low)
            new_high = min(high, range1_high, range2_high)
            if new_low > new_high:
                # No valid range
                print("No")
                return
            low = new_low
            high = new_high
    
    # If we have a valid range, print Yes
    print("Yes")

if __name__ == "__main__":
    main()