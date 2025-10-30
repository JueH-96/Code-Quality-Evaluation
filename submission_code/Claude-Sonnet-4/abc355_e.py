import sys

def solve():
    N, L, R = map(int, input().split())
    
    total_sum = 0
    current = L
    
    while current <= R:
        # Find the largest power of 2 range starting at current that fits in [L, R]
        best_i = -1
        best_j = -1
        
        # Try different values of i (power of 2)
        for i in range(N + 1):
            block_size = 1 << i  # 2^i
            
            # Check if current is aligned to this block size
            if current % block_size != 0:
                continue
                
            # Find j such that the range starts at current
            j = current // block_size
            
            # Check if this query is valid
            if block_size * (j + 1) > (1 << N):
                continue
                
            # Check if the range fits within [L, R]
            range_start = block_size * j
            range_end = block_size * (j + 1) - 1
            
            if range_start == current and range_end <= R:
                best_i = i
                best_j = j
        
        if best_i == -1:
            # If no power-of-2 range works, query individual element
            # This means querying with i=0, j=current
            best_i = 0
            best_j = current
        
        # Make the query
        print(f"? {best_i} {best_j}")
        sys.stdout.flush()
        
        response = int(input())
        if response == -1:
            return
            
        total_sum = (total_sum + response) % 100
        
        # Move to next unprocessed position
        block_size = 1 << best_i
        current = block_size * (best_j + 1)
    
    print(f"! {total_sum}")
    sys.stdout.flush()

solve()