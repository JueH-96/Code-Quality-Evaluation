def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    # Calculate cumulative sum and track the minimum prefix sum.
    current_sum = 0
    min_prefix = 0
    for value in A:
        current_sum += value
        if current_sum < min_prefix:
            min_prefix = current_sum
    
    # The smallest starting number that never makes the passenger count negative.
    required_start = -min_prefix  # if min_prefix is negative, we need to compensate.
    
    # Final number of passengers is starting passengers plus sum of A.
    final_passengers = required_start + current_sum
    print(final_passengers)

if __name__ == '__main__':
    main()