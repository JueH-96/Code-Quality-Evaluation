import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    max_level = 25
    S = [0] * (max_level + 2)  # S[0] to S[25], and S[26]
    
    for k in range(max_level + 1):
        M = 1 << k
        residues = defaultdict(lambda: [0, 0])  # [count, sum]
        for x in A:
            r = x % M
            residues[r][0] += 1
            residues[r][1] += x
        current_S = 0
        processed = set()
        keys = list(residues.keys())
        for r in keys:
            if r in processed:
                continue
            s_complement = (M - r) % M
            if s_complement not in residues:
                continue
            if r < s_complement:
                count_r, sum_r = residues[r]
                count_s, sum_s = residues[s_complement]
                current_S += sum_r * count_s + sum_s * count_r
                processed.add(r)
                processed.add(s_complement)
            elif r == s_complement:
                count_r, sum_r = residues[r]
                current_S += sum_r * (count_r + 1)
                processed.add(r)
        S[k] = current_S
    
    S[max_level + 1] = 0
    total = 0
    for k in range(max_level + 1):
        numerator = S[k] - S[k+1]
        denominator = 1 << k
        total += numerator // denominator
    
    print(total)

if __name__ == "__main__":
    main()