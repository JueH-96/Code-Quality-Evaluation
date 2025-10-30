# YOUR CODE HERE
N, D = map(int, input().split())
S = input()

# Count initial cookies
initial_cookies = S.count('@')

# Calculate remaining cookies after D days
remaining_cookies = initial_cookies - D

# Calculate empty boxes
empty_boxes = N - remaining_cookies

print(empty_boxes)