# YOUR CODE HERE
def find_kth_321_like_number(K):
    from collections import deque

    # Queue for BFS
    queue = deque()
    
    # Start with all single digit numbers (1 to 9)
    for i in range(1, 10):
        queue.append(i)
    
    # Counter for how many 321-like numbers we have generated
    count = 0
    
    # Result variable
    result = None
    
    # BFS to generate numbers
    while queue:
        # Get the current number
        current = queue.popleft()
        count += 1
        
        # If this is the K-th number, we found our answer
        if count == K:
            result = current
            break
        
        # Get the last digit of the current number
        last_digit = current % 10
        
        # Try to append a smaller digit to form a new number
        for next_digit in range(last_digit):
            new_number = current * 10 + next_digit
            queue.append(new_number)
    
    return result

# Read input
import sys
input = sys.stdin.read
K = int(input().strip())

# Find and print the K-th 321-like number
print(find_kth_321_like_number(K))