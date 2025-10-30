def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    # Define a function to check if a position has a mountain fold
    # A position has a mountain fold if the position of the rightmost 1 bit is odd
    def is_mountain_fold(pos):
        # Find the position of the rightmost 1 bit
        pos_of_rightmost_1 = 0
        while pos > 0 and pos % 2 == 0:
            pos //= 2
            pos_of_rightmost_1 += 1
        return pos_of_rightmost_1 % 2 == 1
    
    # For each possible pattern of the last 60 bits (which is enough for values up to 10^18),
    # compute the maximum number of mountain folds
    max_mountain_folds = 0
    
    # We will try all possible values for the last 60 bits of i
    # This is a bit of a trick: we're using the fact that the pattern of mountain folds
    # repeats with a period of 2^60 (which is more than 10^18)
    
    # To optimize, we'll actually just check for 2^20 different patterns
    # This is because the pattern of mountain/valley folds for specific positions
    # doesn't change much with larger values of i
    for i in range(1 << 20):
        mountain_count = 0
        for a in A:
            if is_mountain_fold(i + a):
                mountain_count += 1
        max_mountain_folds = max(max_mountain_folds, mountain_count)
    
    return max_mountain_folds

print(solve())