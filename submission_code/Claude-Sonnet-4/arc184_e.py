def apply_operation(seq):
    """Apply the cumulative XOR operation to a sequence"""
    result = []
    cumsum = 0
    for x in seq:
        cumsum ^= x
        result.append(cumsum)
    return tuple(result)

def find_f(seq1, seq2):
    """Find f(i,j) - minimum operations to make seq1 and seq2 identical"""
    if seq1 == seq2:
        return 0
    
    # Since sequences are binary and finite length, cycles are guaranteed
    # We need at most M operations to find a cycle or solution
    current1, current2 = seq1, seq2
    
    for step in range(1, len(seq1) + 1):
        current1 = apply_operation(current1)
        current2 = apply_operation(current2)
        
        if current1 == current2:
            return step
    
    return 0  # No solution within M steps

# Read input
n, m = map(int, input().split())
sequences = []
for _ in range(n):
    seq = tuple(map(int, input().split()))
    sequences.append(seq)

# Calculate sum of all f(i,j)
total = 0
MOD = 998244353

for i in range(n):
    for j in range(i, n):
        f_val = find_f(sequences[i], sequences[j])
        total = (total + f_val) % MOD

print(total)