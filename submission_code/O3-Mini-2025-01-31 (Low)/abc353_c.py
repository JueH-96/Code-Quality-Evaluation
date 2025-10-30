def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD_VALUE = 10**8

    # Total sum over all pairs without the modulus adjustment
    total_sum = (N - 1) * sum(A)
    
    # Count the number of pairs (i, j) with i < j such that A[i] + A[j] >= 10^8
    A.sort()
    count_adjustments = 0
    left, right = 0, N - 1
    while left < right:
        if A[left] + A[right] >= MOD_VALUE:
            # If for current left, A[left] + A[right] is already >= MOD_VALUE,
            # then for any index between left and right, the sum with A[right] will also be >= MOD_VALUE.
            count_adjustments += (right - left)
            right -= 1
        else:
            left += 1

    # The final answer subtracts 10^8 for each such pair.
    answer = total_sum - count_adjustments * MOD_VALUE
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()