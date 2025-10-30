def main():
    import sys
    import bisect

    input_data = sys.stdin.read().split()
    # Parse input data
    n = int(input_data[0])
    X = list(map(int, input_data[1:n+1]))
    P = list(map(int, input_data[n+1: 2*n+1]))
    q = int(input_data[2*n+1])
    
    # Precompute prefix sums for villagers
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i+1] = prefix[i] + P[i]
    
    result_lines = []
    index = 2*n+2  # starting index for queries in input_data
    
    for _ in range(q):
        L = int(input_data[index])
        R = int(input_data[index+1])
        index += 2
        # find left bound: first index with X[i] >= L
        left_index = bisect.bisect_left(X, L)
        # find right bound: last index with X[i] <= R
        # using bisect_right to get index of first element > R, subtract 1 for inclusive, we sum prefix up to that index.
        right_index = bisect.bisect_right(X, R)
        # Sum villagers in villages from left_index to right_index-1
        total = prefix[right_index] - prefix[left_index]
        result_lines.append(str(total))
    
    sys.stdout.write("
".join(result_lines))

if __name__ == '__main__':
    main()