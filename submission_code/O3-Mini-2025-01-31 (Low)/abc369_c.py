def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    A = list(map(int, data[1:]))

    # The idea is to partition the sequence into maximal contiguous blocks
    # that form an arithmetic progression.
    # In any such block of length L, every subarray is an arithmetic progression,
    # and the number of subarrays from a block of length L is L*(L+1)//2.
    # We just have to be careful that our partitioning covers the entire array.
    
    total = 0
    i = 0
    while i < N:
        # Every single element forms a 1-length arithmetic progression.
        # If it's the last element, just add 1.
        if i == N - 1:
            total += 1
            break
        
        # At the very least, A[i] and A[i+1] always form an arithmetic progression (difference d)
        j = i + 1  # j points to the current element in the running block
        d = A[j] - A[i]
        
        # Extend the block as long as the consecutive differences remain the same.
        # Note: For block lengths of only 0 or 1 indices, it always holds by definition.
        while j + 1 < N and (A[j + 1] - A[j] == d):
            j += 1

        # The block [i, j] is a maximal contiguous arithmetic progression.
        L = j - i + 1
        # Count all subarrays in this block.
        total += L * (L + 1) // 2

        # Move i past the block that we have just processed.
        i = j + 1

    sys.stdout.write(str(total))


if __name__ == '__main__':
    main()