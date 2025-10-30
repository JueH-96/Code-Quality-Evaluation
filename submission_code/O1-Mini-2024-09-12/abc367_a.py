A, B, C = map(int, input().split())

if B < C:
    # Sleeps from B to C
    awake = not (B <= A < C)
else:
    # Sleeps from B to 24 and 0 to C
    awake = not (A >= B or A < C)

print("Yes" if awake else "No")