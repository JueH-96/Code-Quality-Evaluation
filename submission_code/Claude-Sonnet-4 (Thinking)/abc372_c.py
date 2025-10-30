n, q = map(int, input().split())
s = list(input().strip())

# Initial count of ABC patterns
abc_count = 0
for i in range(n - 2):
    if s[i:i+3] == ['A', 'B', 'C']:
        abc_count += 1

for _ in range(q):
    x, c = input().split()
    x = int(x) - 1  # Convert to 0-indexed
    
    # Count old patterns that will be affected
    old_count = 0
    for start_pos in [x-2, x-1, x]:
        if 0 <= start_pos <= n-3:
            if s[start_pos:start_pos+3] == ['A', 'B', 'C']:
                old_count += 1
    
    # Change the character
    s[x] = c
    
    # Count new patterns at the same positions
    new_count = 0
    for start_pos in [x-2, x-1, x]:
        if 0 <= start_pos <= n-3:
            if s[start_pos:start_pos+3] == ['A', 'B', 'C']:
                new_count += 1
    
    # Update total count
    abc_count = abc_count - old_count + new_count
    print(abc_count)