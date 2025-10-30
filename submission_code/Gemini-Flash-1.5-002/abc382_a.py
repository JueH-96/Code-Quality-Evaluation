n, d = map(int, input().split())
s = input()
count = 0
for i in range(n):
    if s[i] == '@':
        count += 1
empty_boxes = 0
cookies_eaten = 0
for i in range(n):
    if s[i] == '.':
        empty_boxes += 1
    else:
        if cookies_eaten < d:
            cookies_eaten += 1
        else:
            empty_boxes += 1

print(empty_boxes)