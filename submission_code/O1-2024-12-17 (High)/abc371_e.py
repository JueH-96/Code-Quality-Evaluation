def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # For each distinct value, we store its positions in the array
    positions = [[] for _ in range(N+1)]
    for i, val in enumerate(A):
        positions[val].append(i+1)  # using 1-based indexing

    # Total number of subranges in an array of length N
    total_subranges = N * (N+1) // 2

    answer = 0

    # For each distinct value val, compute how many subranges it appears in at least once
    for val in range(1, N+1):
        p = positions[val]
        if not p:
            continue  # skip values not present in the array

        # Count the number of subranges that do NOT contain val
        # Break the array into segments between consecutive appearances of val
        sum_no_val = 0

        # Segment before the first occurrence
        left_segment_length = p[0] - 1
        sum_no_val += left_segment_length * (left_segment_length + 1) // 2

        # Segments between consecutive occurrences
        for i in range(len(p) - 1):
            gap_length = p[i+1] - p[i] - 1
            sum_no_val += gap_length * (gap_length + 1) // 2

        # Segment after the last occurrence
        right_segment_length = N - p[-1]
        sum_no_val += right_segment_length * (right_segment_length + 1) // 2

        # Subranges that contain val = total_subranges - subranges that do not contain val
        subranges_with_val = total_subranges - sum_no_val

        # Each subrange that contains val contributes +1 to the distinct count
        answer += subranges_with_val

    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()