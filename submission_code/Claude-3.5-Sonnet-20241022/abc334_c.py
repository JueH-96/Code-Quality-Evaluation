N, K = map(int, input().split())
A = list(map(int, input().split()))

# Create frequency array for socks
freq = [2] * (N+1)
for a in A:
    freq[a] -= 1

# Get list of colors with their counts
colors = []
for i in range(1, N+1):
    if freq[i] > 0:
        colors.extend([i] * freq[i])

# Sort colors
colors.sort()

# Calculate number of pairs we need to make
pairs_needed = (2*N - K) // 2

# Calculate minimum weirdness
total_weirdness = 0
for i in range(pairs_needed):
    # Take 2 socks at a time from sorted list
    # This ensures minimum difference between colors
    if 2*i + 1 < len(colors):
        total_weirdness += abs(colors[2*i] - colors[2*i+1])

print(total_weirdness)