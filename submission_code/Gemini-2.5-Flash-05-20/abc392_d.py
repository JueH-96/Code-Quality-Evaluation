import sys
import collections

def solve():
    # Read N, the number of dice
    N = int(sys.stdin.readline())

    dice_info = []
    # Store information for each die: (K_i, counts_i)
    # K_i is the total number of faces
    # counts_i is a Counter object mapping face values to their frequencies
    for _ in range(N):
        # Read the line for the current die
        # The first element is K_i, the rest are face values
        line = list(map(int, sys.stdin.readline().split()))
        K_i = line[0]
        faces = line[1:]
        
        # Use collections.Counter to efficiently count face frequencies
        counts_i = collections.Counter(faces)
        
        dice_info.append((K_i, counts_i))

    max_prob = 0.0

    # Iterate through all unique pairs of dice (i, j)
    # i goes from 0 to N-2
    # j goes from i+1 to N-1
    for i in range(N):
        for j in range(i + 1, N):
            # Get the K_i and counts_i for die i
            K1, counts1 = dice_info[i]
            # Get the K_j and counts_j for die j
            K2, counts2 = dice_info[j]

            current_prob_numerator_sum = 0

            # To optimize the calculation, iterate over the Counter object
            # that has fewer unique keys. This minimizes the number of dictionary lookups.
            if len(counts1) <= len(counts2):
                # Iterate over counts1, check for values in counts2
                for val, freq1 in counts1.items():
                    if val in counts2:
                        freq2 = counts2[val]
                        current_prob_numerator_sum += (freq1 * freq2)
            else:
                # Iterate over counts2, check for values in counts1
                for val, freq2 in counts2.items():
                    if val in counts1:
                        freq1 = counts1[val]
                        current_prob_numerator_sum += (freq1 * freq2)
            
            # Calculate the probability for this specific pair of dice
            # P(same | i, j) = (1 / (K1 * K2)) * Sum_{v} [ counts1[v] * counts2[v] ]
            # The sum is current_prob_numerator_sum
            current_pair_prob = current_prob_numerator_sum / (K1 * K2)
            
            # Update the maximum probability found so far
            max_prob = max(max_prob, current_pair_prob)

    # Print the result with sufficient precision
    print(f"{max_prob:.15f}")

# Call the solve function to run the program
solve()