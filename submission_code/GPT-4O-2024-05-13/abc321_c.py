def generate_321_like_numbers():
    from collections import deque
    
    # Initialize a deque with single digit numbers
    queue = deque([str(i) for i in range(1, 10)])
    result = []
    
    while queue:
        num = queue.popleft()
        result.append(int(num))
        
        # Get the last digit of the current number
        last_digit = int(num[-1])
        
        # Generate new numbers by appending smaller digits
        for next_digit in range(last_digit - 1, -1, -1):
            new_num = num + str(next_digit)
            queue.append(new_num)
    
    return result

def find_kth_321_like_number(K):
    # Generate all 321-like numbers
    numbers = generate_321_like_numbers()
    # Sort the numbers to ensure they are in ascending order
    numbers.sort()
    # Return the K-th smallest number
    return numbers[K - 1]

# Read input
import sys
input = sys.stdin.read
K = int(input().strip())

# Find and print the K-th smallest 321-like number
print(find_kth_321_like_number(K))