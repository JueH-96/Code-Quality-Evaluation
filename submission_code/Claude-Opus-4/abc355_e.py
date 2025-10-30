import sys

def find_blocks(L, R):
    """Find the blocks needed to cover the range [L, R]"""
    blocks = []
    pos = L
    
    while pos <= R:
        # Find the largest i such that pos is divisible by 2^i
        # and pos + 2^i - 1 <= R
        best_i = -1
        for i in range(20):  # 2^20 > 2^18, so this is enough
            if pos % (1 << i) == 0 and pos + (1 << i) - 1 <= R:
                best_i = i
            else:
                break
        
        if best_i == -1:
            # This shouldn't happen if our logic is correct
            break
            
        # j = pos / 2^i
        j = pos >> best_i
        blocks.append((best_i, j))
        pos += (1 << best_i)
    
    return blocks

def main():
    # Read input
    N, L, R = map(int, input().split())
    
    # Find the blocks we need to query
    blocks = find_blocks(L, R)
    
    # Query each block and sum the results
    total = 0
    for i, j in blocks:
        # Send query
        print(f"? {i} {j}")
        sys.stdout.flush()
        
        # Read response
        response = int(input())
        if response == -1:
            # Error occurred
            return
        
        total = (total + response) % 100
    
    # Output the answer
    print(f"! {total}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()