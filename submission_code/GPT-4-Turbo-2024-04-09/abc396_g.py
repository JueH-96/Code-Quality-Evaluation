import sys
input = sys.stdin.read

def solve():
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    grid = data[2:]
    
    # Since W is small (up to 18), we can use bitmasking to represent column flips
    min_sum = float('inf')
    
    # There are 2^W possible ways to flip columns
    for col_mask in range(1 << W):
        # Create a list to count how many rows will have each possible pattern after column flips
        pattern_count = {}
        
        for row in grid:
            # Calculate the new row pattern after applying the column flips
            new_pattern = 0
            for j in range(W):
                original_value = int(row[j])
                # If we are flipping this column, invert the value
                if col_mask & (1 << j):
                    original_value = 1 - original_value
                # Set the bit in the new pattern
                new_pattern = (new_pattern << 1) | original_value
            
            if new_pattern in pattern_count:
                pattern_count[new_pattern] += 1
            else:
                pattern_count[new_pattern] = 1
        
        # Now, for each possible row pattern, we can either take it as is or flip it entirely
        current_min = 0
        for pattern, count in pattern_count.items():
            # Bits in pattern tell us how many 1s are there if not flipped
            num_ones = bin(pattern).count('1')
            num_zeros = W - num_ones
            # If we flip the entire row, the number of 1s becomes the number of zeros
            # We want to minimize the number of 1s
            best_for_pattern = min(num_ones, num_zeros) * count
            current_min += best_for_pattern
        
        # Update the global minimum
        min_sum = min(min_sum, current_min)
    
    print(min_sum)