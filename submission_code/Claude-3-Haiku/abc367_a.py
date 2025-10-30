# Read the input
a, b, c = map(int, input().split())

# Check if Takahashi can shout his love for takoyaki every day
if a in range(b, c):
    print("Yes")
else:
    print("No")