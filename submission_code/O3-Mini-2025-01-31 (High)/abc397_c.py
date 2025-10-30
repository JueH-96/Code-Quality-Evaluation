def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # Compute prefix distinct counts for the subarray A[0..i]
    prefix = [0] * N
    seen_left = set()
    distinct_count = 0
    for i in range(N):
        if A[i] not in seen_left:
            seen_left.add(A[i])
            distinct_count += 1
        prefix[i] = distinct_count

    # Compute suffix distinct counts for the subarray A[i..N-1]
    suffix = [0] * N
    seen_right = set()
    distinct_count = 0
    for i in range(N - 1, -1, -1):
        if A[i] not in seen_right:
            seen_right.add(A[i])
            distinct_count += 1
        suffix[i] = distinct_count

    # Try all valid split positions and maximize the sum of distinct counts
    max_sum = 0
    for i in range(N - 1):
        current_sum = prefix[i] + suffix[i + 1]
        if current_sum > max_sum:
            max_sum = current_sum

    sys.stdout.write(str(max_sum))

if __name__ == '__main__':
    main()