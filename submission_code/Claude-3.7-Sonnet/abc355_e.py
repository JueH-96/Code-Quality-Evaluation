def solve():
    N, L, R = map(int, input().split())
    
    total_remainder = 0
    
    while L <= R:
        # Find the largest i such that L is a multiple of 2^i
        # and the segment [L, L + 2^i - 1] is within [L, R]
        # and 2^i * (j+1) <= 2^N
        i = 0
        while True:
            next_i = i + 1
            if (L % (1 << next_i) == 0 and 
                L + (1 << next_i) - 1 <= R and 
                (L // (1 << next_i)) + 1 <= (1 << (N - next_i))):
                i = next_i
            else:
                break
        
        # Compute the corresponding j
        j = L // (1 << i)
        
        # Ask the judge
        print(f"? {i} {j}", flush=True)
        
        # Read the remainder from the judge
        remainder = int(input())
        
        # If we got an error, terminate the program
        if remainder == -1:
            exit(0)
        
        # Add the remainder to the total
        total_remainder = (total_remainder + remainder) % 100
        
        # Move to the next chunk
        L += (1 << i)
    
    # Output the final result
    print(f"! {total_remainder}", flush=True)

if __name__ == "__main__":
    solve()