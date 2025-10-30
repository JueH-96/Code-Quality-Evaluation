def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    X = int(data[1])
    # Read the sequence A and record original indices (1-indexed)
    A = list(map(int, data[2:2+n]))
    arr = [(A[i], i + 1) for i in range(n)]
    
    # Sort the array by value to allow a two pointer method
    arr.sort(key=lambda x: x[0])
    
    # For each fixed element, use two pointers to search for a pair that sums up to (X - current)
    for i in range(n - 2):
        a_val, a_idx = arr[i]
        left = i + 1
        right = n - 1
        while left < right:
            b_val, b_idx = arr[left]
            c_val, c_idx = arr[right]
            s = a_val + b_val + c_val
            if s == X:
                # We have found a valid triple, sort indices in increasing order and print them.
                indices = sorted([a_idx, b_idx, c_idx])
                sys.stdout.write(f"{indices[0]} {indices[1]} {indices[2]}")
                return
            elif s < X:
                left += 1
            else:
                right -= 1
    sys.stdout.write("-1")
    
if __name__ == '__main__':
    main()