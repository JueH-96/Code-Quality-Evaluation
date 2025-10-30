def is_good_sequence(l, r):
    if l >= r:
        return False
    
    # Find minimum i such that 2^i > l
    i = 0
    while (1 << i) <= l:
        i += 1
    i -= 1
    
    # Check if sequence can be represented as S(2^i*j, 2^i*(j+1))
    if l % (1 << i) == 0:
        j = l // (1 << i)
        if r == (j + 1) * (1 << i):
            return True
    
    return False

def find_next_good_sequence(l, r):
    # Try different powers of 2
    i = 0
    while (1 << i) <= l:
        i += 1
    i -= 1
    
    # Try different j values
    while i >= 0:
        j = l // (1 << i)
        next_r = (j + 1) * (1 << i)
        
        if next_r > l and next_r <= r and is_good_sequence(l, next_r):
            return next_r
        i -= 1
    
    return l + 1

def solve(L, R):
    result = []
    current = L
    
    while current < R:
        next_pos = find_next_good_sequence(current, R)
        result.append((current, next_pos))
        current = next_pos
    
    return result

# Read input
L, R = map(int, input().split())

# Get solution
sequences = solve(L, R)

# Print output
print(len(sequences))
for l, r in sequences:
    print(l, r)