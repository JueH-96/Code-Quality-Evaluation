import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    ptr += M

    # Preprocess the sorted list of (A[i], i) sorted by A[i], then i
    sorted_A = []
    for i in range(N):
        sorted_A.append((A[i], i))
    sorted_A.sort()

    # Extract the keys for binary search
    keys = [a for a, i in sorted_A]

    # Precompute min_indices
    min_indices = []
    min_so_far = float('inf')
    for a, i in sorted_A:
        min_so_far = min(min_so_far, i)
        min_indices.append(min_so_far)

    # Process each query
    for b in B:
        # Find the rightmost index where keys[k] <= b
        k = bisect.bisect_right(keys, b) - 1
        if k < 0:
            print(-1)
        else:
            earliest_i = min_indices[k]
            print(earliest_i + 1)

if __name__ == "__main__":
    main()