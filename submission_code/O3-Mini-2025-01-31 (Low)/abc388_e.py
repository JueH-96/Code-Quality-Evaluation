def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))
    # The array is already sorted in ascending order.
    # We'll use a two pointers approach:
    # Use left pointer (for the smaller top mochi) starting from index 0,
    # and right pointer (for the bottom mochi) starting from the middle of the array.
    left = 0
    mid = N // 2
    right = mid
    count = 0
    
    # We only consider left pointers in the first half (indices 0 to mid-1)
    # and we try to pair each with a bottom mochi from the right half if it satisfies
    # the condition: top*2 <= bottom.
    while left < mid and right < N:
        # If the size of mochi at left (top) * 2 is less or equal to mochi at right (bottom)
        if A[left] * 2 <= A[right]:
            count += 1
            left += 1
            right += 1
        else:
            right += 1

    sys.stdout.write(str(count))

if __name__ == '__main__':
    main()