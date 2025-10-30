n = int(input())
s = input().strip()
t = input().strip()
count = 0
for a, b in zip(s, t):
    if a != b:
        count += 1
print(count)