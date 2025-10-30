import sys

# Read N
N = int(sys.stdin.readline())

dice_freq = []
dice_K = []

# Read dice information
for _ in range(N):
    line = list(map(int, sys.stdin.readline().split()))
    K = line[0]
    faces = line[1:]

    freq = {}
    for face in faces:
        freq[face] = freq.get(face, 0) + 1

    dice_K.append(K)
    dice_freq.append(freq)

max_prob = 0.0

# Iterate through all pairs of distinct dice (i, j)
for i in range(N):
    for j in range(i + 1, N):
        freq_i = dice_freq[i]
        K_i = dice_K[i]
        freq_j = dice_freq[j]
        K_j = dice_K[j]

        current_prob = 0.0

        # To calculate sum_x [ (count_i(x) / K_i) * (count_j(x) / K_j) ]
        # Iterate through the frequency map with fewer unique faces
        # This optimizes the loop iterations.
        # We sum over face_val present in the iterated dict.
        # freq_j.get(face_val, 0) gives count_j(face_val) or 0 if not present.
        # The term is non-zero only if face_val is present in both dictionaries.
        if len(freq_i) <= len(freq_j):
            for face_val, count_i in freq_i.items():
                count_j = freq_j.get(face_val, 0)
                current_prob += (count_i / float(K_i)) * (count_j / float(K_j))
        else: # len(freq_j) < len(freq_i)
            for face_val, count_j in freq_j.items():
                count_i = freq_i.get(face_val, 0)
                current_prob += (count_i / float(K_i)) * (count_j / float(K_j))

        max_prob = max(max_prob, current_prob)

# Print the result with required precision
print(f"{max_prob:.15f}")