A, B = map(int, input().split())

def is_arithmetic(p, q, r):
    return q - p == r - q

def check_sequence(x, A, B):
    # Check all possible arrangements
    seqs = [
        (x, A, B),
        (x, B, A),
        (A, x, B),
        (A, B, x),
        (B, A, x),
        (B, x, A)
    ]
    
    return any(is_arithmetic(p, q, r) for p, q, r in seqs)

count = 0
# Since A,B ≤ 100, x must be within a reasonable range to form arithmetic sequence
for x in range(-200, 300):
    if check_sequence(x, A, B):
        count += 1

print(count)