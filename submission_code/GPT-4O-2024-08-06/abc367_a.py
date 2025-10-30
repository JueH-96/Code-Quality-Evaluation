# YOUR CODE HERE
def can_shout_love_for_takoyaki(A, B, C):
    if B < C:
        # Awake from C to B (next day)
        if A < B or A >= C:
            return "Yes"
        else:
            return "No"
    else:
        # Awake from C to B (spanning over midnight)
        if A >= C or A < B:
            return "Yes"
        else:
            return "No"

# Read inputs
A, B, C = map(int, input().split())

# Determine if Takahashi can shout his love for takoyaki
result = can_shout_love_for_takoyaki(A, B, C)

# Print the result
print(result)