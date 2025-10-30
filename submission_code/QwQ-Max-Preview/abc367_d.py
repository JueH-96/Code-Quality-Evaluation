import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    
    # Compute prefix sums
    S = [0] * (N + 1)
    for i in range(1, N + 1):
        S[i] = S[i-1] + A[i-1]
    
    X = S[N] % M
    
    # Case 1: s < t, count pairs where S[t-1] ≡ S[s-1] mod M
    residues = defaultdict(int)
    for i in range(N):
        mod = S[i] % M
        residues[mod] += 1
    case1 = 0
    for cnt in residues.values():
        case1 += cnt * (cnt - 1) // 2
    
    # Case 2: t < s, count pairs where (S[t-1] - S[s-1] + X) mod M == 0
    # Which is equivalent to S[s-1] ≡ (S[t-1] + X) mod M
    freq = defaultdict(int)
    case2 = 0
    for i in reversed(range(N)):
        current_residue = (S[i] + X) % M
        case2 += freq.get(current_residue, 0)
        mod = S[i] % M
        freq[mod] += 1
    
    print(case1 + case2)

if __name__ == '__main__':
    main()