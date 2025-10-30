def count_empty_boxes_after_days(N, D, S):
    # Count the number of cookies in the string S
    cookie_count = S.count('@')
    
    # Calculate the number of boxes that will be empty after D days
    # Since we know there are at least D cookies, we can simply subtract D from N
    empty_boxes = N - (cookie_count - D)
    
    return empty_boxes

# Read input
import sys
input = sys.stdin.read
data = input().strip().splitlines()

N, D = map(int, data[0].split())
S = data[1]

# Get the result
result = count_empty_boxes_after_days(N, D, S)

# Print the result
print(result)