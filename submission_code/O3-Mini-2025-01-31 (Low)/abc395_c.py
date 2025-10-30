def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    # We will use a dictionary to record the last seen index for each number.
    last_index = {}
    min_length = n + 1  # initialize with a large number
    for i in range(n):
        num = A[i]
        if num in last_index:
            # the subarray from the previous occurrence (last_index[num]) to i has a repeated value.
            current_length = i - last_index[num] + 1
            min_length = min(min_length, current_length)
        last_index[num] = i
    
    # If min_length was updated, we print it; otherwise, no valid subarray exists.
    print(min_length if min_length != n + 1 else -1)
    
if __name__ == "__main__":
    main()