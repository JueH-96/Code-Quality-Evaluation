n = int(input())
from collections import defaultdict

# Group slimes by their odd part
groups = defaultdict(list)

for _ in range(n):
    s, c = map(int, input().split())
    
    # Express s as odd_part * 2^power
    odd_part = s
    power = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power += 1
    
    groups[odd_part].append((power, c))

total_slimes = 0

for odd_part, power_counts in groups.items():
    # For this odd part, collect counts at each power of 2
    counts = defaultdict(int)
    for power, count in power_counts:
        counts[power] += count
    
    # Perform optimal synthesis using binary addition with carries
    carry = 0
    max_power = max(counts.keys())
    power = 0
    
    while carry > 0 or power <= max_power:
        total_count = counts[power] + carry
        total_slimes += total_count % 2  # Remaining slimes at this power
        carry = total_count // 2  # Slimes carried to the next power
        power += 1

print(total_slimes)