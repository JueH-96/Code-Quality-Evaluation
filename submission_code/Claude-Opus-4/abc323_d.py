# YOUR CODE HERE
n = int(input())
slimes = {}

for _ in range(n):
    s, c = map(int, input().split())
    slimes[s] = c

# Keep combining slimes until no more combinations possible
while True:
    combined = False
    
    # Find a size with at least 2 slimes
    sizes_to_check = list(slimes.keys())
    
    for size in sizes_to_check:
        if size in slimes and slimes[size] >= 2:
            # Combine pairs of slimes of this size
            pairs = slimes[size] // 2
            remaining = slimes[size] % 2
            
            # Update the count for current size
            if remaining > 0:
                slimes[size] = remaining
            else:
                del slimes[size]
            
            # Add the combined slimes of size 2*size
            new_size = 2 * size
            if new_size in slimes:
                slimes[new_size] += pairs
            else:
                slimes[new_size] = pairs
            
            combined = True
            break
    
    if not combined:
        break

# Count total remaining slimes
total = sum(slimes.values())
print(total)