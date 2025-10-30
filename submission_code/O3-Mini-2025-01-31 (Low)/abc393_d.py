def main():
    import sys
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    s = input_data[1].strip()

    # Collect the positions (0-indexed) of all '1's in the string.
    ones_pos = []
    for index, char in enumerate(s):
        if char == '1':
            ones_pos.append(index)
    
    # Let k be the number of ones.
    k = len(ones_pos)
    
    # The goal is to move all ones so that they are contiguous.
    # The optimal strategy is to make the block of ones centered around
    # the median of their original positions.
    #
    # If the ones are placed at positions ones_pos[0], ones_pos[1], ..., ones_pos[k-1],
    # then we imagine that if they become contiguous then the new positions must be:
    # base, base+1, ... base+k-1
    # The best base is to choose it such that the median remains in the same relative position.
    # As a result, using the median index `mid = k//2`, we choose:
    # base = ones_pos[mid] - mid
    #
    # Then, the number of adjacent swaps required is the sum
    # of absolute differences between current positions and target positions:
    #   Sum(|ones_pos[i] - (base + i)|) for i in 0..k-1
    #
    # This works because an adjacent swap effectively moves an element by one,
    # and the sum count of moves is the number of swaps needed.
    
    mid = k // 2
    base = ones_pos[mid] - mid
    swaps = 0
    for i in range(k):
        swaps += abs(ones_pos[i] - (base + i))
    
    sys.stdout.write(str(swaps))

if __name__ == '__main__':
    main()