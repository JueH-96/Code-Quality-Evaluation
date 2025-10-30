def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    if N == 1:
        print(-1)
        return
    
    # Dictionary to store the last index of each element
    last_index = {}
    min_length = float('inf')
    
    for i in range(N):
        if A[i] in last_index:
            # Calculate the length of the subarray
            length = i - last_index[A[i]] + 1
            min_length = min(min_length, length)
        
        # Update the last index of the current element
        last_index[A[i]] = i
    
    if min_length == float('inf'):
        print(-1)
    else:
        print(min_length)

if __name__ == "__main__":
    main()