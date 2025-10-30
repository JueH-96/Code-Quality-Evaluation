def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # If there's only one element, the answer is 1 (the single-element subarray).
    if N == 1:
        print(1)
        return

    # We'll use a variable 'length' to keep track of the length of the
    # longest arithmetic progression suffix ending at the current position.
    # 'diff' will store the common difference of that suffix.

    answer = 1  # (A[0] by itself is an AP)
    length = 1
    diff = None

    for i in range(1, N):
        current_diff = A[i] - A[i - 1]
        if current_diff == diff:
            length += 1
        else:
            length = 2  # Restart a new AP suffix of length 2
            diff = current_diff
        answer += length

    print(answer)

# Don't forget to call main()
main()