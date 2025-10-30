# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    N = int(data[0])
    L = int(data[1])
    R = int(data[2])
    
    total_sum_mod_100 = 0
    
    # We will use a greedy approach to cover the range [L, R] with power-of-two segments
    current = L
    while current <= R:
        # Find the largest power of two segment that starts at or after `current` and ends at or before `R`
        max_power = 1
        while max_power <= R - current:
            max_power *= 2
        max_power //= 2
        
        # Check if we can use this power of two starting from `current`
        if current % max_power == 0:
            # This means we can use the segment [current, current + max_power - 1]
            l = current
            r = current + max_power - 1
        else:
            # Otherwise, we need to adjust to the next power of two boundary
            l = (current // max_power) * max_power
            r = l + max_power - 1
            if r > R:
                max_power //= 2
                l = (current // max_power) * max_power
                r = l + max_power - 1
        
        # Ask the query for the segment [l, r]
        i = (r - l + 1).bit_length() - 1
        j = l // (1 << i)
        
        print(f"? {i} {j}", flush=True)
        response = int(input().strip())
        
        if response == -1:
            return
        
        total_sum_mod_100 = (total_sum_mod_100 + response) % 100
        
        # Move to the next segment
        current = r + 1
    
    # Output the final result
    print(f"! {total_sum_mod_100}", flush=True)

if __name__ == "__main__":
    main()