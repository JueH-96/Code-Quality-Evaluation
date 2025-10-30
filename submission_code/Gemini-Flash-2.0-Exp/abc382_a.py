N, D = map(int, input().split())
S = input()

cookies = S.count('@')
empty_boxes = N - (cookies - D)

print(empty_boxes)