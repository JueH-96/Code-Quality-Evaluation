import sys

def solve():
    # Read N, the number of dice
    N = int(sys.stdin.readline())
    
    dice_data = []
    for _ in range(N):
        line = list(map(int, sys.stdin.readline().split()))
        K = line[0]      # Number of faces for the current die
        faces = line[1:] # List of numbers on the faces
        
        # Create a frequency map for the faces of the current die
        freq_map = {}
        for face_val in faces:
            freq_map[face_val] = freq_map.get(face_val, 0) + 1
        
        dice_data.append({'K': K, 'freq': freq_map})

    max_prob = 0.0

    # Iterate over all distinct pairs of dice
    for i in range(N):
        for j in range(i + 1, N): # Ensures j > i, so pairs are distinct and not repeated
            # Data for the first die in the pair (die 'a')
            die_a_data = dice_data[i]
            K_a = die_a_data['K']
            freq_a = die_a_data['freq']
            
            # Data for the second die in the pair (die 'b')
            die_b_data = dice_data[j]
            K_b = die_b_data['K']
            freq_b = die_b_data['freq']

            current_sum_counts_product = 0

            # To calculate sum_{v} [ count_a(v) * count_b(v) ],
            # iterate over the face values present on one die and check against the other.
            # For a minor optimization, iterate over the die with fewer distinct face values.
            
            map_to_iterate = freq_a
            map_to_lookup = freq_b
            
            if len(freq_a) > len(freq_b):
                # If freq_a has more distinct faces than freq_b,
                # swap them so we iterate over the smaller map (freq_b).
                map_to_iterate = freq_b
                map_to_lookup = freq_a

            for val, count_in_iter_map in map_to_iterate.items():
                # val is a face value, count_in_iter_map is its frequency on the iterated die.
                # Get its frequency on the other die (0 if not present).
                count_in_lookup_map = map_to_lookup.get(val, 0)
                current_sum_counts_product += count_in_iter_map * count_in_lookup_map
            
            # Probability = sum_counts_product / (K_a * K_b)
            # K_a and K_b are guaranteed to be >= 1 by problem constraints (1 <= K_i).
            # So K_a * K_b will not be zero.
            # In Python 3, / performs float division.
            current_prob = float(current_sum_counts_product) / (K_a * K_b)
            
            if current_prob > max_prob:
                max_prob = current_prob
                
    # Print the maximum probability found, formatted to 15 decimal places.
    # This matches sample output style and typical competitive programming precision requirements.
    print(f"{max_prob:.15f}")

# Standard boilerplate to call the main function.
if __name__ == '__main__':
    solve()