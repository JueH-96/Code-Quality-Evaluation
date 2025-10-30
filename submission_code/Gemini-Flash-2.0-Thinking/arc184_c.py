def solve():
    n = int(input())
    a = list(map(int, input().split()))

    def is_mountain(index):
        if index <= 0:
            return False
        count = 0
        temp = index
        while temp > 0 and temp % 2 == 0:
            count += 1
            temp //= 2
        return count % 2 == 1

    max_mountain_folds = 0
    num_creases = 2**100 - 1

    # Instead of iterating through all possible i, we can consider the relative positions
    # and the structure of the mountain fold sequence.

    # The pattern of mountain folds is related to the binary representation of the index.
    # The i-th crease is a mountain fold if the number of trailing zeros in the binary
    # representation of i is odd.

    # We are looking for a contiguous block of creases where the pattern of mountain
    # folds matches the offsets given by A.

    # Consider the differences between consecutive elements of A.

    max_f = 0
    upper_bound = 2**16 # A sufficiently large number to capture the pattern
    for start in range(1, upper_bound):
        current_f = 0
        for k in range(n):
            if is_mountain(start + a[k]):
                current_f += 1
        max_f = max(max_f, current_f)

    print(max_f)

solve()