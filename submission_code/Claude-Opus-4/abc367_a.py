# YOUR CODE HERE
A, B, C = map(int, input().split())

if B < C:
    # Sleeps during the same day from B to C
    # Awake if A < B or A >= C
    if A < B or A >= C:
        print("Yes")
    else:
        print("No")
else:
    # Sleeps overnight from B to C (next day)
    # Awake if C <= A < B
    if C <= A < B:
        print("Yes")
    else:
        print("No")