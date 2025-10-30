def main():
    import sys
    import math
    
    input_data = sys.stdin.read().strip().split()
    N, L, R = map(int, input_data[:3])
    
    # Helper function to send a query and get the answer.
    # Returns the sum (mod 100) of A in the specified block [2^i * j, 2^i * (j+1) - 1].
    def ask(i, j):
        # Print the query in the required format
        print(f"? {i} {j}", flush=True)
        # Read the judge's response
        T = int(sys.stdin.readline())
        # If T == -1, something went wrong or constraints not satisfied,
        # so we must terminate immediately
        if T < 0:
            sys.exit(0)
        return T
    
    # This function returns a list of (i, j) pairs of power-of-two aligned blocks
    # that exactly cover the interval [start, end] without overlap.
    def cover_interval(start, end):
        blocks = []
        left = start
        right = end
        
        # Cover from the left side
        while left <= right:
            block_size = left & -left  # largest power of two dividing 'left'
            # Check if we can take that full block without exceeding "right"
            if left + block_size - 1 <= right:
                i = int(math.log2(block_size))
                j = left >> i
                blocks.append((i, j))
                left += block_size
            else:
                break
        
        # Cover from the right side
        while left <= right:
            block_size = (right + 1) & -(right + 1)  # largest power of two dividing (right+1)
            next_left = right - block_size + 1
            if next_left >= left:
                i = int(math.log2(block_size))
                j = next_left >> i
                blocks.append((i, j))
                right -= block_size
            else:
                break
        
        return blocks
    
    # Get the list of blocks needed to cover [L, R]
    blocks_to_query = cover_interval(L, R)
    
    # Sum up the answers mod 100
    answer_mod_100 = 0
    for (i, j) in blocks_to_query:
        answer_mod_100 = (answer_mod_100 + ask(i, j)) % 100
    
    # Output the final answer and flush
    print(f"! {answer_mod_100}", flush=True)

# Don't forget to call main() to actually run the solution.
if __name__ == "__main__":
    main()