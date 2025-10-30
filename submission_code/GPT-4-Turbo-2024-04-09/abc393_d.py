def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    # Collect all the positions of '1' in the string
    positions = [i for i, char in enumerate(S) if char == '1']
    
    # Total number of 1s
    total_ones = len(positions)
    
    # The optimal position to gather all 1s is around the median of the positions of 1s
    # If we gather all 1s around the median index, it minimizes the total distance
    median_index = positions[total_ones // 2]
    
    # Calculate the minimum number of swaps needed to make all 1s contiguous around the median
    min_swaps = 0
    for i, pos in enumerate(positions):
        # Target position for the i-th '1' should be around the median
        target_pos = median_index - (total_ones // 2) + i
        min_swaps += abs(pos - target_pos)
    
    print(min_swaps)

if __name__ == "__main__":
    main()