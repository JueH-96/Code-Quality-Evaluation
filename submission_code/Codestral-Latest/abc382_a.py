# YOUR CODE HERE
N, D = map(int, input().split())
S = input()

# Count the number of cookies in the boxes
cookie_count = S.count('@')

# Calculate the number of empty boxes after D days
empty_boxes = N - cookie_count

print(empty_boxes)