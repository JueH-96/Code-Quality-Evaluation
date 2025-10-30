def fold_type(index):
    """ Determine the type of fold at a given index using binary representation logic.
        '1' in binary representation indicates a mountain fold (M), '0' indicates a valley fold (V).
    """
    return bin(index).count('1') % 2  # 0 for valley (even number of 1s), 1 for mountain (odd number of 1s)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Since we need to maximize the number of mountain folds in the sequence
    # We need to find the maximum number of 1s in the binary representation of (i + A_k) for all k
    max_mountain_folds = 0
    
    # We will use a dictionary to count occurrences of each pattern
    fold_count = {}
    
    # We only need to consider indices up to 2^100 - A_N - 1
    # But we cannot iterate up to 2^100, instead we will use the properties of the folds
    # We will iterate over possible i values that are relevant given the maximum A_N
    max_i = (1 << 100) - A[-1] - 1
    
    # We calculate fold types for i + A_k for each i and count the maximum number of mountains
    for i in range(max_i + 1):
        current_fold_count = 0
        for a in A:
            if fold_type(i + a):
                current_fold_count += 1
        max_mountain_folds = max(max_mountain_folds, current_fold_count)
    
    print(max_mountain_folds)

if __name__ == "__main__":
    main()