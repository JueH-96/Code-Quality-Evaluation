# YOUR CODE HERE
def find_shortest_subarray_with_repeated_value():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Dictionary to store the last seen index of each element
    last_seen = {}
    min_length = float('inf')
    
    for i in range(N):
        if A[i] in last_seen:
            # Calculate the length of the subarray
            subarray_length = i - last_seen[A[i]] + 1
            min_length = min(min_length, subarray_length)
        
        # Update the last seen index of the current element
        last_seen[A[i]] = i
    
    # If min_length was updated, return it, otherwise return -1
    if min_length == float('inf'):
        print(-1)
    else:
        print(min_length)

find_shortest_subarray_with_repeated_value()