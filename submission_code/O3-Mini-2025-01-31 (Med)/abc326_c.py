def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    gifts = [int(next(it)) for _ in range(N)]
    
    # Sort the gifts' coordinates for a sliding window approach
    gifts.sort()
    
    max_count = 0
    right = 0  # right pointer for the window
    
    # Use the left pointer to iterate over each potential starting gift
    for left in range(N):
        # Increase right pointer as long as the gift is within [gifts[left], gifts[left]+M)
        while right < N and gifts[right] < gifts[left] + M:
            right += 1
        # Gifts in the interval are from index left to right-1, count is right - left.
        max_count = max(max_count, right - left)
        
    sys.stdout.write(str(max_count))

if __name__ == '__main__':
    main()