# Read input
N, D = map(int, input().split())
S = input().strip()

# Count total cookies
total_cookies = S.count('@')

# After D days, D cookies will be eaten
# So remaining cookies will be total_cookies - D
remaining_cookies = total_cookies - D

# Empty boxes will be total boxes minus remaining cookies
empty_boxes = N - remaining_cookies

# Print result
print(empty_boxes)