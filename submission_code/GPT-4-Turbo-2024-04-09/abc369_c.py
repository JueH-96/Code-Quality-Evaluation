def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    if N == 1:
        print(1)
        return
    
    total_count = 0
    start = 0
    
    while start < N:
        end = start + 1
        if end == N:
            total_count += 1
            break
        
        # Calculate the common difference
        d = A[end] - A[start]
        
        while end < N and A[end] - A[end - 1] == d:
            end += 1
        
        # Length of the arithmetic progression segment
        length = end - start
        
        # Count of all subarrays starting from start to end-1
        # Sum of first 'length' natural numbers: length * (length + 1) // 2
        total_count += length * (length + 1) // 2
        
        # Move start to the end of the current arithmetic progression segment
        start = end - 1
    
    print(total_count)

if __name__ == "__main__":
    main()