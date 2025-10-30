# Read input
N, T, A = map(int, input().split())
remaining = N - (T + A)

# Check if either candidate has already secured a win
if T > A + remaining or A > T + remaining:
    print("Yes")
else:
    print("No")