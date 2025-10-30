# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Sort the array to find the minimum difference
    A_sorted = sorted(A)
    
    min_diff = float('inf')
    
    # We need to choose a window of size (N - K) in the sorted array
    window_size = N - K
    for i in range(len(A_sorted) - window_size + 1):
        current_diff = A_sorted[i + window_size - 1] - A_sorted[i]
        if current_diff < min_diff:
            min_diff = current_diff
    
    print(min_diff)

if __name__ == "__main__":
    main()