def find_nth_good_integer(N):
    # The good digits are 0, 2, 4, 6, 8
    good_digits = ['0', '2', '4', '6', '8']
    
    # We will use a BFS approach to generate numbers using these digits
    from collections import deque
    
    queue = deque(good_digits[1:])  # Start with non-zero digits for the first position
    count = 0  # To count the numbers generated
    
    # Special case for N = 1, the smallest good integer is 0
    if N == 1:
        return 0
    
    # We have already skipped '0' so start count from 1
    count = 1
    
    while queue:
        current = queue.popleft()
        
        # Each time we pop from the queue, we have found the next smallest good integer
        if count == N:
            return int(current)
        
        # Generate new numbers by appending each good digit to the current number
        for digit in good_digits:
            new_number = current + digit
            queue.append(new_number)
        
        count += 1

# Read input from standard input (usually for competitive programming)
import sys
input = sys.stdin.read
N = int(input().strip())

# Find the N-th good integer
result = find_nth_good_integer(N)

# Print the result to standard output
print(result)