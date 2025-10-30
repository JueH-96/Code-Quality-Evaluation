# YOUR CODE HERE
L, R = map(int, input().split())

result = []
current = L

while current < R:
    if current == 0:
        # Find the largest power of 2 that is <= R
        max_power = 1
        while max_power * 2 <= R:
            max_power *= 2
    else:
        # Find the largest power of 2 that divides current
        max_power = current & -current
        
    # Ensure that current + max_power <= R
    while current + max_power > R:
        max_power //= 2
    
    # Add the good sequence
    result.append((current, current + max_power))
    current += max_power

print(len(result))
for l, r in result:
    print(l, r)