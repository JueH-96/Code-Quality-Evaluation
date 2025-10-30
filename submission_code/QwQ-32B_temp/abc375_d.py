import sys
from collections import defaultdict

def main():
    S = sys.stdin.readline().strip()
    char_indices = defaultdict(list)
    n = len(S)
    for i in range(n):
        char = S[i]
        pos = i + 1  # 1-based index
        char_indices[char].append(pos)
    
    total = 0
    for indices in char_indices.values():
        m = len(indices)
        if m < 2:
            continue
        # Compute prefix sums
        prefix = [0] * (m + 1)
        for i in range(m):
            prefix[i + 1] = prefix[i] + indices[i]
        current_sum = 0
        for j in range(1, m):
            term = indices[j] * j - prefix[j] - j
            current_sum += term
        total += current_sum
    print(total)

if __name__ == "__main__":
    main()