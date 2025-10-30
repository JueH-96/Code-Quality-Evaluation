N, A, B = map(int, input().split())
D = list(map(int, input().split()))

week_length = A + B
D_mod = [d % week_length for d in D]

found = False
for start in range(min(week_length, 10**6)):
    valid = True
    for d in D_mod:
        if (start + d) % week_length >= A:
            valid = False
            break
    if valid:
        found = True
        break

print("Yes" if found else "No")