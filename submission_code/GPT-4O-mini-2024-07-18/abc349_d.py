def divide_into_good_sequences(L, R):
    result = []
    current = L
    
    while current < R:
        # Find the largest power of 2 that is less than or equal to current
        power_of_2 = 1
        while power_of_2 <= current:
            power_of_2 <<= 1
        
        # The previous power of 2 is the one we want
        power_of_2 >>= 1
        
        # Calculate j such that l = 2^i * j
        j = current // power_of_2
        
        # Calculate the next good sequence
        l = power_of_2 * j
        r = power_of_2 * (j + 1)
        
        # If r exceeds R, we need to adjust it to R
        if r > R:
            r = R
        
        # Add the current good sequence to the result
        result.append((l, r))
        
        # Move to the next starting point
        current = r
    
    return result

def main():
    import sys
    input = sys.stdin.read
    L, R = map(int, input().strip().split())
    
    good_sequences = divide_into_good_sequences(L, R)
    
    print(len(good_sequences))
    for l, r in good_sequences:
        print(l, r)

if __name__ == "__main__":
    main()