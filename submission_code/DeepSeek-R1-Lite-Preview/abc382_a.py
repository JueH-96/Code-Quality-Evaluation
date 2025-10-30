N, D = map(int, input().split())
S = input().strip()

# Count the number of '@' in S
C = S.count('@')

# Calculate the number of remaining cookies
remaining_cookies = C - D

# Calculate the number of empty boxes
empty_boxes = N - remaining_cookies

print(empty_boxes)