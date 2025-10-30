def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    idx = 0
    N = int(input_data[idx])
    idx += 1
    dices = []
    # For each die, get K and then the faces, count occurrences.
    for _ in range(N):
        K = int(input_data[idx])
        idx += 1
        faces = list(map(int, input_data[idx: idx + K]))
        idx += K
        freq = {}
        for face in faces:
            freq[face] = freq.get(face, 0) + 1
        dices.append((K, freq)) # store (total faces, frequency dictionary)
        
    best = 0.0
    # Iterate over all pair of dices
    for i in range(N):
        K_i, freq_i = dices[i]
        for j in range(i+1, N):
            K_j, freq_j = dices[j]
            prob = 0.0
            # iterate over keys of the smaller dictionary to speed up if possible.
            if len(freq_i) > len(freq_j):
                freq_i, freq_j = freq_j, freq_i
                K_i, K_j = K_j, K_i
            for face, count in freq_i.items():
                if face in freq_j:
                    prob += (count / K_i) * (freq_j[face] / K_j)
            if prob > best:
                best = prob
                
    print(best)
    
if __name__ == '__main__':
    main()