n, x = map(int, input().split())
a = list(map(int, input().split()))

min_score = 101  # Initialize with a value higher than the maximum possible S (100)

for s in range(0, 101):
    # Create a new list including the current score s
    b = a + [s]
    b.sort()
    # Calculate the sum of B[1] to B[N-2] (0-based)
    total = sum(b[1 : len(b) - 1])
    if total >= x:
        if s < min_score:
            min_score = s

# Check if a valid score was found
if min_score <= 100:
    print(min_score)
else:
    print(-1)