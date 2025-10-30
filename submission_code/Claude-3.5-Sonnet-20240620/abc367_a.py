# YOUR CODE HERE
def can_shout_love(A, B, C):
    if B < C:  # Normal sleep schedule (e.g., 22:00 to 06:00)
        return A < B or A >= C
    else:  # Sleep schedule crosses midnight (e.g., 22:00 to 06:00)
        return A < B and A >= C

A, B, C = map(int, input().split())
print("Yes" if can_shout_love(A, B, C) else "No")