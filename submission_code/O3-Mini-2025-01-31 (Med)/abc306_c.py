def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))

    # To store the count of occurrences for each i and the middle occurrence index.
    count = [0] * (N + 1)
    mid_index = [0] * (N + 1)
    
    # For each occurrence in A, record the index when it's the second (middle) occurrence.
    for idx, x in enumerate(A):
        count[x] += 1
        if count[x] == 2:
            mid_index[x] = idx

    # Create a list of numbers 1 to N and sort them by their respective f(i) (i.e., mid_index)
    nums = list(range(1, N + 1))
    nums.sort(key=lambda x: mid_index[x])
    
    # Print the result
    sys.stdout.write(" ".join(map(str, nums)))
    
if __name__ == '__main__':
    main()