N, M = map(int, input().split())
H = list(map(int, input().split()))

count = 0
for h in H:
    if M >= h:
        M -= h
        count += 1
    else:
        break

print(count)