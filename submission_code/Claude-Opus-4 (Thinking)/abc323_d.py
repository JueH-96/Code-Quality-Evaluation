from collections import defaultdict

n = int(input())
slimes = defaultdict(int)

for _ in range(n):
    s, c = map(int, input().split())
    slimes[s] = c

# Keep combining until no more combinations possible
while True:
    new_slimes = defaultdict(int)
    changed = False
    
    for size, count in slimes.items():
        if count >= 2:
            changed = True
            pairs = count // 2
            remaining = count % 2
            
            if remaining > 0:
                new_slimes[size] += remaining
            new_slimes[size * 2] += pairs
        else:
            new_slimes[size] += count
    
    slimes = new_slimes
    if not changed:
        break

# Count total slimes
total = sum(slimes.values())
print(total)