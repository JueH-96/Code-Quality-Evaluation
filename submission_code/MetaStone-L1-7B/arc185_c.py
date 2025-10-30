def find_triplet_with_sum():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    X = int(input[1])
    A = list(map(int, input[2:2+N]))
    
    for j in range(1, N-1):
        target = X - A[j]
        # Elements before j
        before = A[:j]
        # Elements after j
        after = A[j+1:]
        # Check if target - a_i exists in after
        for a_i in before:
            if (target - a_i) in after:
                # Find the indices
                idx_i = before.index(a_i)
                idx_k = after.index(target - a_i)
                print(idx_i + 1, idx_j + 1, idx_k + 1)
                return
    print(-1)

find_triplet_with_sum()