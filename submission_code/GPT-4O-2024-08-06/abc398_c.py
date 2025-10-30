# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    from collections import Counter
    
    # Count the occurrences of each integer
    count = Counter(A)
    
    # Find the maximum unique integer
    max_unique_value = -1
    max_unique_index = -1
    
    for i in range(N):
        if count[A[i]] == 1:
            if A[i] > max_unique_value:
                max_unique_value = A[i]
                max_unique_index = i + 1  # +1 because labels are 1-based
    
    # Output the result
    if max_unique_index == -1:
        print(-1)
    else:
        print(max_unique_index)

if __name__ == "__main__":
    main()