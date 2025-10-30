def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    n = int(data[0])
    A = list(map(int, data[1:]))

    last_index = {}
    # Set answer to a very large number initially.
    ans = float('inf')

    for i, val in enumerate(A):
        if val in last_index:
            # Calculate the length of the subarray from the previous index of val to i.
            length = i - last_index[val] + 1
            if length < ans:
                ans = length
        # Update the last seen index for the value.
        last_index[val] = i

    # If no qualifying subarray found, ans remains infinity.
    if ans == float('inf'):
        sys.stdout.write(str(-1))
    else:
        sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()