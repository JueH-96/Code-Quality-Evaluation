import sys
from collections import Counter

def main():
    """
    This program solves the dice probability problem by following these steps:
    1. Reads the number of dice, N.
    2. For each die, it reads the number of faces (K) and the values on those faces.
       It uses a collections.Counter to efficiently store the frequency of each face value.
    3. It iterates through all unique pairs of dice.
    4. For each pair, it calculates the probability that they show the same number.
    5. It keeps track of the maximum probability found among all pairs.
    6. Finally, it prints the maximum probability with the required precision.
    """

    # Read N, the number of dice.
    try:
        N_str = sys.stdin.readline()
        if not N_str: return
        N = int(N_str)
    except (IOError, ValueError):
        return

    # Store data for each die: its total number of faces (K) and a Counter
    # object holding the frequency of each face value.
    dice_data = []
    for _ in range(N):
        line = list(map(int, sys.stdin.readline().split()))
        K = line[0]
        faces = line[1:]
        counts = Counter(faces)
        dice_data.append((K, counts))

    # This will store the maximum probability found.
    max_prob = 0.0

    # Iterate through all unique pairs of dice (i, j) where i < j.
    for i in range(N):
        for j in range(i + 1, N):
            # Get the pre-processed data for the two dice in the pair.
            K_i, counts_i = dice_data[i]
            K_j, counts_j = dice_data[j]

            # The probability of rolling the same number is:
            # (1 / (K_i * K_j)) * sum over all values v [count_i(v) * count_j(v)]
            # First, calculate the sum part.
            numerator_sum = 0
            
            # For efficiency, iterate over the keys of the smaller Counter.
            if len(counts_i) > len(counts_j):
                smaller_counts, larger_counts = counts_j, counts_i
            else:
                smaller_counts, larger_counts = counts_i, counts_j

            # Sum the product of counts for each face value present in the smaller die.
            for value, count_in_smaller in smaller_counts.items():
                count_in_larger = larger_counts.get(value, 0)
                numerator_sum += count_in_smaller * count_in_larger
            
            # The denominator is the total number of outcomes for the pair.
            denominator = K_i * K_j
            
            # Calculate the probability for the current pair.
            # Constraints guarantee K_i >= 1, so no division by zero.
            current_prob = numerator_sum / denominator

            # Update the maximum probability found so far.
            if current_prob > max_prob:
                max_prob = current_prob
                
    # Print the final result with high precision to meet error tolerance.
    print(f"{max_prob:.12f}")

if __name__ == "__main__":
    main()