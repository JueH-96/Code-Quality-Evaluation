# YOUR CODE HERE
M = int(input())
S1 = input().strip()
S2 = input().strip()
S3 = input().strip()

# Precompute the positions for each character in each reel
from collections import defaultdict

# Create a dictionary to map each character to the list of positions it appears in each reel
char_to_positions = defaultdict(lambda: [[], [], []])

for i, c in enumerate(S1):
    char_to_positions[c][0].append(i)
for i, c in enumerate(S2):
    char_to_positions[c][1].append(i)
for i, c in enumerate(S3):
    char_to_positions[c][2].append(i)

min_time = float('inf')

# Iterate over all possible characters
for c in char_to_positions:
    positions1 = char_to_positions[c][0]
    positions2 = char_to_positions[c][1]
    positions3 = char_to_positions[c][2]
    
    if not positions1 or not positions2 or not positions3:
        continue
    
    # Iterate over all possible combinations of positions
    for p1 in positions1:
        for p2 in positions2:
            for p3 in positions3:
                # Calculate the time for each reel to stop
                # The time is the maximum of the times when each reel is stopped
                # The time for each reel is t_i = p_i + k_i * M, where k_i is the number of full cycles
                # We need to find t such that t >= p1, t >= p2, t >= p3, and t is the smallest possible
                # Since t must be the same for all reels, we need to find t such that t ≡ p1 mod M, t ≡ p2 mod M, t ≡ p3 mod M
                # This is equivalent to finding t such that t ≡ p1 mod M, t ≡ p2 mod M, t ≡ p3 mod M
                # Since M is the same for all reels, the only way this is possible is if p1 ≡ p2 ≡ p3 mod M
                # So, we need to find t such that t ≡ p1 mod M, t ≡ p2 mod M, t ≡ p3 mod M
                # Which is only possible if p1 ≡ p2 ≡ p3 mod M
                if p1 % M == p2 % M == p3 % M:
                    # The earliest time is the maximum of p1, p2, p3
                    t = max(p1, p2, p3)
                    if t < min_time:
                        min_time = t

if min_time != float('inf'):
    print(min_time)
else:
    print(-1)