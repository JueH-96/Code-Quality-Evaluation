A, B, C = map(int, input().split())

if B < C:
    # Normal case: sleeps from B to C
    # Awake from 0 to B and from C to 24
    if (0 <= A < B) or (C <= A < 24):
        print("Yes")
    else:
        print("No")
else:
    # Sleeps across midnight: sleeps from B to 24 and from 0 to C
    # Awake from C to B
    if C <= A < B:
        print("Yes")
    else:
        print("No")