# YOUR CODE HERE
A, B, C = map(int, input().split())

if B < C:
    # Sleeps from B to C on the same day
    if A < B or A >= C:
        print("Yes")
    else:
        print("No")
else:
    # B > C, sleeps from B today to C tomorrow
    if C <= A < B:
        print("Yes")
    else:
        print("No")