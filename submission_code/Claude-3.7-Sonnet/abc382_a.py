# Read input
N, D = map(int, input().split())
S = input().strip()

# Count the initial number of boxes with cookies
cookies_count = S.count('@')

# Calculate the number of empty boxes after D days
# Initially, (N - cookies_count) boxes are empty
# After eating D cookies, D more boxes will become empty
empty_boxes = N - cookies_count + D

# Print the result
print(empty_boxes)