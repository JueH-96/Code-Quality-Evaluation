def main():
    import sys
    from collections import Counter

    data = sys.stdin.read().split()
    if not data:
        return

    idx = 0
    N = int(data[idx])
    idx += 1

    # Prepare a list of tuples: (counter of faces, number of faces)
    dice = []
    for _ in range(N):
        K = int(data[idx])
        idx += 1
        # Read the K face values
        faces = list(map(int, data[idx:idx+K]))
        idx += K
        freq = Counter(faces)
        dice.append((freq, K))

    best_probability = 0.0

    # Compare every pair of dice
    for i in range(N):
        freq1, K1 = dice[i]
        for j in range(i + 1, N):
            freq2, K2 = dice[j]
            # Intersection of face values that appear on both dice
            common_faces = freq1.keys() & freq2.keys()
            probability = 0.0
            for face in common_faces:
                # For the face, probability of showing it in dice1 is freq1[face] / K1
                # and in dice2 is freq2[face] / K2; multiply to get the combined probability.
                probability += (freq1[face] / K1) * (freq2[face] / K2)
            if probability > best_probability:
                best_probability = probability

    # Output the answer. The precision requirement is taken care of by the print output.
    print(best_probability)

if __name__ == '__main__':
    main()