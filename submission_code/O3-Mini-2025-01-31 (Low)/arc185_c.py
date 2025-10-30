def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    X = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # We'll work with a list of (value, original_index)
    arr = [(A[i], i+1) for i in range(n)]
    arr.sort(key=lambda x: x[0])
    
    # We use a two pointer approach for each fixed i
    for i in range(n - 2):
        a_i, idx_i = arr[i]
        target = X - a_i
        left = i + 1
        right = n - 1
        while left < right:
            a_left, idx_left = arr[left]
            a_right, idx_right = arr[right]
            two_sum = a_left + a_right
            if two_sum == target:
                # We need to output triple in increasing order of original indices,
                # but note that problem demands i < j < k with respect to the positions in the input.
                # However, after sorting by value, the original indices might be in arbitrary order.
                # So we reorder them.
                triple = [idx_i, idx_left, idx_right]
                triple.sort()
                # Check if they satisfy i < j < k.
                # They will because we sorted the triple in increasing order.
                print(triple[0], triple[1], triple[2])
                return
            elif two_sum < target:
                left += 1
            else:
                right -= 1
    print(-1)

if __name__ == '__main__':
    main()