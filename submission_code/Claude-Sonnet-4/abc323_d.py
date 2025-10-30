from collections import defaultdict

n = int(input())
slimes = defaultdict(int)

for _ in range(n):
    s, c = map(int, input().split())
    slimes[s] = c

# Keep synthesizing until no more synthesis is possible
changed = True
while changed:
    changed = False
    new_slimes = defaultdict(int)
    
    for size, count in slimes.items():
        if count >= 2:
            # Perform synthesis
            remaining = count % 2
            synthesized = count // 2
            
            if remaining > 0:
                new_slimes[size] += remaining
            if synthesized > 0:
                new_slimes[size * 2] += synthesized
                changed = True
        else:
            # Can't synthesize, keep as is
            new_slimes[size] += count
    
    slimes = new_slimes

# Count total remaining slimes
total = sum(slimes.values())
print(total)