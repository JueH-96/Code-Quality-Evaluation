n, k = map(int, input().split())
lost = list(map(int, input().split()))

# Count remaining socks of each color
sock_count = [2] * (n + 1)  # sock_count[i] = number of socks of color i
sock_count[0] = 0  # no color 0

# Remove lost socks
for color in lost:
    sock_count[color] -= 1

# First, make pairs within same colors (weirdness = 0)
unpaired_socks = []
for color in range(1, n + 1):
    pairs_from_this_color = sock_count[color] // 2
    remaining = sock_count[color] % 2
    
    # Add remaining unpaired socks to the list
    for _ in range(remaining):
        unpaired_socks.append(color)

# Sort unpaired socks by color
unpaired_socks.sort()

# Greedily pair adjacent socks in sorted order
total_weirdness = 0
pairs_to_make = len(unpaired_socks) // 2

for i in range(pairs_to_make):
    sock1 = unpaired_socks[2 * i]
    sock2 = unpaired_socks[2 * i + 1]
    total_weirdness += abs(sock1 - sock2)

print(total_weirdness)