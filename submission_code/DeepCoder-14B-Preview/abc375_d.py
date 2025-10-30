from collections import defaultdict

def count_palindromic_triples(s):
    char_positions = defaultdict(list)
    for idx, c in enumerate(s):
        pos = idx + 1  # Convert to 1-based index
        char_positions[c].append(pos)
    
    total = 0
    for c in char_positions:
        pos_list = char_positions[c]
        m = len(pos_list)
        if m < 2:
            continue
        
        # Compute the prefix sum array
        prefix = [0] * (m + 1)
        for i in range(1, m + 1):
            prefix[i] = prefix[i - 1] + pos_list[i - 1]
        
        # Calculate contributions for each j
        for j in range(1, m):
            contribution = j * pos_list[j] - prefix[j] - j
            total += contribution
    
    return total

# Read input
s = input().strip()
# Compute and print the result
print(count_palindromic_triples(s))