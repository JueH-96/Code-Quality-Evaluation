B = int(input())

result = -1
for A in range(1, 21):
    power = A ** A
    if power == B:
        result = A
        break
    if power > B:
        break

print(result)