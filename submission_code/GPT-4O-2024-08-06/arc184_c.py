# YOUR CODE HERE
def count_ones(x):
    return bin(x).count('1')

def is_mountain(x):
    return count_ones(x) % 2 == 1

def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Precompute if each A_k is a mountain fold
    mountain_folds = [is_mountain(A_k) for A_k in A]
    
    # We need to find the maximum number of mountain folds in any contiguous subsequence
    max_mountains = 0
    current_mountains = 0
    
    # Sliding window over the sequence A
    for i in range(N):
        if mountain_folds[i]:
            current_mountains += 1
        else:
            current_mountains = 0
        
        max_mountains = max(max_mountains, current_mountains)
    
    print(max_mountains)

if __name__ == "__main__":
    main()