def transform_sequence(seq):
    M = len(seq)
    new_seq = [0] * M
    for i in range(M):
        # Calculate prefix sum mod 2
        new_seq[i] = sum(seq[:i+1]) % 2
    return tuple(new_seq)

def find_cycle_and_steps(seq1, seq2, M):
    if seq1 == seq2:
        return 0
    
    # Keep track of seen states for seq1
    seen = {seq1: 0}
    current = seq1
    steps = 0
    
    while True:
        steps += 1
        current = transform_sequence(current)
        
        if current == seq2:
            return steps
        
        if current in seen:
            # If we found a cycle and haven't matched seq2, it's impossible
            return 0
            
        if steps > M:  # If we haven't found a match after M steps, it's impossible
            return 0
            
        seen[current] = steps

def main():
    MOD = 998244353
    
    # Read input
    N, M = map(int, input().split())
    sequences = []
    for _ in range(N):
        seq = tuple(map(int, input().split()))
        sequences.append(seq)
    
    # Calculate sum of f(i,j) for all i <= j
    total = 0
    for i in range(N):
        for j in range(i, N):
            result = find_cycle_and_steps(sequences[i], sequences[j], M)
            total = (total + result) % MOD
    
    print(total)

if __name__ == "__main__":
    main()