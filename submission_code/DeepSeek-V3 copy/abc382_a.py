# YOUR CODE HERE
N, D = map(int, input().split())
S = input().strip()
# Count the number of '@' in S
total_cookies = S.count('@')
# After D days, the number of cookies left is total_cookies - D
cookies_left = total_cookies - D
# The number of empty boxes is N minus the number of boxes that still have cookies
empty_boxes = N - cookies_left
print(empty_boxes)