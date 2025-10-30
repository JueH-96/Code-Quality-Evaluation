def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    # First value is N
    n = int(input_data[0])
    # Next n values are the sequence A
    a = list(map(int, input_data[1:]))

    # Dictionary to store the last index of each value.
    last_index = {}
    min_length = float('inf')

    # Iterate through the sequence.
    for i, num in enumerate(a):
        if num in last_index:
            # Compute subarray length from the last occurrence to current index.
            current_length = i - last_index[num] + 1
            if current_length < min_length:
                min_length = current_length
        # Update the last seen index for num.
        last_index[num] = i

    # If no valid subarray is found, output -1; otherwise, output the shortest length.
    if min_length == float('inf'):
        print(-1)
    else:
        print(min_length)

if __name__ == '__main__':
    main()