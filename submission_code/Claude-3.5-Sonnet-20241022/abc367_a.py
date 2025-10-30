A, B, C = map(int, input().split())

# Function to check if time x is between start and end (considering 24h cycle)
def is_between(x, start, end):
    if start <= end:
        return start <= x <= end
    else:  # When period crosses midnight
        return x >= start or x <= end

# If B <= C, Takahashi is awake from C to B
# If B > C, Takahashi is awake from C to B (next day)
if is_between(A, C, B if B > C else B+24):
    print("Yes")
else:
    print("No")