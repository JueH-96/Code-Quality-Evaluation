N, D = map(int, input().split())
S = input().strip()

count_cookies = S.count('@')
empty_boxes = N - count_cookies + D

print(empty_boxes)