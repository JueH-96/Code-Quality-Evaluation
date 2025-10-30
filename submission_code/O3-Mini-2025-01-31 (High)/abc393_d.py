def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1].strip()
    
    # Collect the positions (0-indexed) of all '1's
    ones_positions = []
    for index, ch in enumerate(S):
        if ch == '1':
            ones_positions.append(index)
    
    # Let k be the number of ones.
    k = len(ones_positions)
    # Choose the median index in the list. This median minimizes the sum of absolute differences.
    mid_index = k // 2
    median = ones_positions[mid_index]
    
    # The idea:
    # We want all ones to be contiguous. Suppose in the final configuration
    # the ones occupy consecutive positions. A natural target is to align
    # the original ones such that the median one stays fixed. Then the target positions
    # for ones would be "median - mid_index, median - mid_index + 1, ..., median - mid_index + k - 1"
    # The number of swaps needed to move a '1' from its original position to the target position
    # is the absolute difference between the two positions. Thus, the answer is:
    #   sum(|ones_positions[i] - (median - mid_index + i)|) for i in 0...k-1
    moves = 0
    for i, pos in enumerate(ones_positions):
        moves += abs(pos - (median - mid_index + i))
    
    sys.stdout.write(str(moves))
    
if __name__ == "__main__":
    main()