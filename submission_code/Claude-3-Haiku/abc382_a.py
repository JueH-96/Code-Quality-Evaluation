# Read input
N, D = map(int, input().split())
S = input()

# Count the number of cookies
cookies = S.count('@')

# Determine the number of empty boxes after D days
empty_boxes = N - min(cookies, D)

# Print the answer
print(empty_boxes)