B = int(input())
for a in range(1, 61):
    current = a ** a
    if current == B:
        print(a)
        exit()
    elif current > B:
        break
print(-1)