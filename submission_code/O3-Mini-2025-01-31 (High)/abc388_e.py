def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    n = int(input_data[0])
    # The sizes are sorted in ascending order.
    A = list(map(int, input_data[1:]))

    # We'll use a two-pointer approach.
    # We need to form pairs (A[i], A[j]) where A[i] is the top and A[j] is the base;
    # the condition being A[i] <= A[j] / 2, or equivalently, A[i] * 2 <= A[j].
    #
    # We split the array roughly into two halves:
    #   - left part for potential top mochi (leaning towards smaller sizes)
    #   - right part for potential base mochi.
    #
    # Initialize left pointer to the first half (starting at index 0)
    # and right pointer to the middle (n // 2).
    left = 0
    right = n // 2
    count = 0

    # While there is candidate in the left side and we have not run out of the right side.
    while left < n // 2 and right < n:
        # Check if current left candidate can be placed on current right candidate.
        if A[left] * 2 <= A[right]:
            count += 1  # valid pair found
            left += 1
            right += 1
        else:
            # Current right candidate is too small. Try next base candidate.
            right += 1

    sys.stdout.write(str(count))

if __name__ == '__main__':
    main()