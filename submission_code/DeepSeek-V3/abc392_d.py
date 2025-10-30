import sys
from collections import defaultdict

def solve():
    input_lines = sys.stdin.read().splitlines()
    N = int(input_lines[0])
    dice = []
    for line in input_lines[1:N+1]:
        parts = list(map(int, line.split()))
        K_i = parts[0]
        faces = parts[1:]
        freq = defaultdict(int)
        for num in faces:
            freq[num] += 1
        dice.append((K_i, freq))
    
    max_prob = 0.0
    for i in range(N):
        K_i, freq_i = dice[i]
        for j in range(i+1, N):
            K_j, freq_j = dice[j]
            common_numbers = set(freq_i.keys()) & set(freq_j.keys())
            prob = 0.0
            for num in common_numbers:
                prob += (freq_i[num] / K_i) * (freq_j[num] / K_j)
            if prob > max_prob:
                max_prob = prob
    print("{0:.15f}".format(max_prob))

solve()