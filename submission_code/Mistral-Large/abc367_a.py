import sys
input = sys.stdin.read
data = input().split()

A = int(data[0])
B = int(data[1])
C = int(data[2])

# Check if Takahashi is awake at time A
if B < C:
    # Normal case where he sleeps and wakes up on the same day
    if B <= A < C:
        print("No")
    else:
        print("Yes")
else:
    # Case where he sleeps past midnight
    if A < C or A >= B:
        print("Yes")
    else:
        print("No")