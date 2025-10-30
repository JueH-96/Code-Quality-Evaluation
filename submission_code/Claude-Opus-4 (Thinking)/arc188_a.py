# YOUR CODE HERE
n, k = map(int, input().split())
s = input().strip()

MOD = 998244353

# Find positions of ? in the string
question_positions = []
for i in range(n):
    if s[i] == '?':
        question_positions.append(i)

num_questions = len(question_positions)
total_ways = 0

# Try all possible replacements (3^num_questions possibilities)
for mask in range(3**num_questions):
    # Convert mask to character replacement
    temp_s = list(s)
    temp_mask = mask
    for i in range(num_questions):
        choice = temp_mask % 3
        temp_mask //= 3
        # Replace ? with A, B, or C based on choice
        temp_s[question_positions[i]] = 'ABC'[choice]
    
    # Count good substrings in this configuration
    good_count = 0
    for i in range(n):
        count_a = 0
        count_b = 0
        count_c = 0
        for j in range(i, n):
            # Count characters in substring s[i:j+1]
            if temp_s[j] == 'A':
                count_a += 1
            elif temp_s[j] == 'B':
                count_b += 1
            else:
                count_c += 1
            
            # Check if substring is good (all counts have same parity)
            if count_a % 2 == count_b % 2 == count_c % 2:
                good_count += 1
    
    # If this configuration has at least k good substrings, count it
    if good_count >= k:
        total_ways = (total_ways + 1) % MOD

print(total_ways)