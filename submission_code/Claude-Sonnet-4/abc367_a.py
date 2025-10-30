# YOUR CODE HERE
A, B, C = map(int, input().split())

# Check if A is during sleeping time
if B < C:
    # Normal case: sleeps from B to C on same day
    is_asleep = B <= A < C
else:
    # Cross midnight case: sleeps from B to 24, then 0 to C
    is_asleep = (A >= B) or (A < C)

if is_asleep:
    print("No")
else:
    print("Yes")