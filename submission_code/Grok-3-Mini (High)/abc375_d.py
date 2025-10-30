import sys

# Read input from stdin
data = sys.stdin.read().strip()
S = data
N = len(S)

# Create a list of lists to store positions for each character (A-Z)
char_positions = [[] for _ in range(26)]

# Store 1-based positions for each character
for idx in range(N):  # idx is 0-based index in string
    char_code = ord(S[idx]) - ord('A')  # Get index 0-25 for A-Z
    pos = idx + 1  # Convert to 1-based position
    char_positions[char_code].append(pos)

# Initialize total answer
total_answer = 0

# For each character group
for pos_list in char_positions:
    m = len(pos_list)  # Number of occurrences of this character
    if m < 2:
        continue  # No pairs if less than 2 occurrences
    
    # Compute the sum using the formula
    sum_part = 0
    for idx in range(m):  # idx is 0-based index in pos_list
        i = idx + 1  # Convert to 1-based index for the formula
        pos_val = pos_list[idx]  # The position value (1-based)
        sum_part += pos_val * (2 * i - 1 - m)
    
    # Binomial coefficient C(m, 2) = m*(m-1)/2
    binom = (m * (m - 1)) // 2
    
    # Group sum for this character
    group_sum = sum_part - binom
    
    # Add to total answer
    total_answer += group_sum

# Output the result
print(total_answer)