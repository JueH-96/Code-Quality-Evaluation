# YOUR CODE HERE
def max_sum_sequence():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Initialize the first element
    current_max_sum = A[0]
    global_max_sum = A[0]
    
    # Iterate over the array starting from the second element
    for i in range(1, N):
        # Calculate the new current max sum
        current_max_sum = max(current_max_sum + A[i], A[i])
        # Update the global max sum
        global_max_sum = max(global_max_sum, current_max_sum)
    
    print(global_max_sum)