def max_mountain_folds(N, A):
    # The maximum number of mountain folds we can have
    max_folds = 0
    
    # We will iterate through all possible values of i from 1 to 2^100 - A[N-1] - 1
    # However, we don't need to actually compute all values of i, we can just analyze the folds.
    
    # The number of folds is determined by the binary representation of the index.
    # Each crease can be represented as a binary digit where:
    # - 0 represents a valley fold
    # - 1 represents a mountain fold
    
    # The number of mountain folds for a specific i is determined by the number of 1s in the binary representation
    # of the index (i + A_k) for each A_k in A.
    
    # We need to find the maximum number of mountain folds for any i in the range.
    
    # The maximum number of mountain folds occurs when we have the most significant bits set to 1.
    # This happens when we consider the binary representation of the numbers.
    
    # The maximum number of mountain folds is determined by the number of bits in the binary representation
    # of the maximum crease index, which is 2^100 - 1.
    
    # The maximum number of mountain folds we can get is equal to the number of A_k that can contribute
    # to the mountain folds.
    
    # We will calculate the maximum number of mountain folds for each A_k.
    
    for a in A:
        # We need to find how many bits are set in (i + a) for i from 1 to 2^100 - a - 1
        # This is equivalent to counting how many bits are set in the range of numbers.
        
        # The maximum number of mountain folds is determined by the number of bits in the binary representation
        # of the maximum crease index.
        # The maximum crease index is 2^100 - 1 - a.
        max_index = (1 << 100) - 1 - a
        
        # Count the number of 1s in the binary representation of max_index
        # This gives us the maximum number of mountain folds for this A_k
        mountain_folds = bin(max_index).count('1')
        
        # Update the maximum folds
        max_folds = max(max_folds, mountain_folds)
    
    return max_folds

import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    result = max_mountain_folds(N, A)
    print(result)

if __name__ == "__main__":
    main()