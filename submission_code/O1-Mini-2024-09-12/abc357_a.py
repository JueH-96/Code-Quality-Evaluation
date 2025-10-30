# Read input from stdin
N, M = map(int, input().split())
H = list(map(int, input().split()))

count = 0
for h in H:
    if M >= h:
        count += 1
        M -= h
    else:
        M = 0
        break

print(count)