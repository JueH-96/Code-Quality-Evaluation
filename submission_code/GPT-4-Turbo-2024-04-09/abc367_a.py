A, B, C = map(int, input().split())

if B < C:
    # Normal sleep schedule within the same day
    if B <= A < C:
        print("No")
    else:
        print("Yes")
else:
    # Sleep schedule wraps around midnight
    if C <= A < B:
        print("Yes")
    else:
        print("No")