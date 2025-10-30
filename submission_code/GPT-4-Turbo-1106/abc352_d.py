from collections import deque

def find_min_good_index_sequence(N, K, P):
    # Create a dictionary to store the index of each element in P
    index_dict = {value: index for index, value in enumerate(P)}
    
    # Initialize a deque to keep track of the current window of K elements
    current_window = deque()
    
    # Initialize the minimum difference to the maximum possible value
    min_diff = N
    
    # Iterate through the permutation P
    for value in P:
        # Append the current value to the window
        current_window.append(value)
        
        # If the window size exceeds K, pop the leftmost element
        if len(current_window) > K:
            current_window.popleft()
        
        # If the window size is K, check if it's a good index sequence
        if len(current_window) == K:
            # Find the minimum and maximum values in the current window
            min_value = min(current_window)
            max_value = max(current_window)
            
            # Check if the current window is a consecutive sequence of integers
            if max_value - min_value == K - 1:
                # Calculate the indices of the min and max values
                min_index = index_dict[min_value]
                max_index = index_dict[max_value]
                
                # Update the minimum difference if a smaller one is found
                min_diff = min(min_diff, max_index - min_index)
    
    # Return the minimum difference
    return min_diff

# Read input from stdin
N, K = map(int, input().split())
P = list(map(int, input().split()))

# Find and print the minimum value of i_K - i_1 among all good index sequences
print(find_min_good_index_sequence(N, K, P) - 1)