MOD = 998244353

def apply_operation(seq):
    n = len(seq)
    new_seq = [0] * n
    cum_sum = 0
    for i in range(n):
        cum_sum = (cum_sum + seq[i]) % 2
        new_seq[i] = cum_sum
    return new_seq

def compute_f(seq1, seq2):
    if seq1 == seq2:
        return 0
    
    # Convert to tuples for efficient comparison
    curr1 = tuple(seq1)
    curr2 = tuple(seq2)
    
    seen = set()
    seen.add((curr1, curr2))
    
    # Maximum possible cycle length is 2^M but we limit iterations
    max_iterations = min(2**len(seq1), 5000)
    
    for x in range(1, max_iterations):
        curr1 = tuple(apply_operation(curr1))
        curr2 = tuple(apply_operation(curr2))
        
        if curr1 == curr2:
            return x
        
        if (curr1, curr2) in seen:
            # We've entered a cycle without finding equality
            return 0
        
        seen.add((curr1, curr2))
    
    return 0

# Read input
N, M = map(int, input().split())
sequences = []
for _ in range(N):
    sequences.append(list(map(int, input().split())))

# Compute the sum
total = 0
for i in range(N):
    for j in range(i, N):
        total = (total + compute_f(sequences[i], sequences[j])) % MOD

print(total)