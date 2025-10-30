def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    
    # Build a list of dice, each dice is a tuple (K, freq_dict)
    dice = []
    for _ in range(N):
        K = int(next(it))
        freq = {}
        for _ in range(K):
            num = int(next(it))
            freq[num] = freq.get(num, 0) + 1
        dice.append((K, freq))
        
    best_probability = 0.0
    # Evaluate every pair of dice
    for i in range(N):
        K_i, freq_i = dice[i]
        for j in range(i+1, N):
            K_j, freq_j = dice[j]
            pair_prob = 0.0
            # iterate through the smaller dictionary for efficiency
            if len(freq_i) <= len(freq_j):
                for face, count_i in freq_i.items():
                    if face in freq_j:
                        pair_prob += (count_i / K_i) * (freq_j[face] / K_j)
            else:
                for face, count_j in freq_j.items():
                    if face in freq_i:
                        pair_prob += (count_j / K_j) * (freq_i[face] / K_i)
            best_probability = max(best_probability, pair_prob)
            
    # Print the best probability with sufficient precision.
    sys.stdout.write("{:.12f}".format(best_probability))

if __name__ == '__main__':
    main()