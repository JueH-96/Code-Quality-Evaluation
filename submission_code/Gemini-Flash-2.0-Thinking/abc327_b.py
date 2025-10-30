B = int(input())

found = False
for A in range(1, 17):
    power = pow(A, A)
    if power == B:
        print(A)
        found = True
        break

if not found:
    print(-1)