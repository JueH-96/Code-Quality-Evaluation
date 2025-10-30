A, B = map(int, input().split())
count = 0
if (A + B) % 2 == 0:
    count += 1
if A != B:
    count += 2
print(count)