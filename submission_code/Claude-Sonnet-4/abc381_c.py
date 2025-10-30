# YOUR CODE HERE
n = int(input())
s = input().strip()

max_length = 0

# Find all positions of '/'
for i in range(n):
    if s[i] == '/':
        # Count consecutive '1's to the left
        left_ones = 0
        j = i - 1
        while j >= 0 and s[j] == '1':
            left_ones += 1
            j -= 1
        
        # Count consecutive '2's to the right
        right_twos = 0
        j = i + 1
        while j < n and s[j] == '2':
            right_twos += 1
            j += 1
        
        # The length of valid 11/22 string centered at this '/'
        # We can take min(left_ones, right_twos) characters from each side
        valid_pairs = min(left_ones, right_twos)
        current_length = 1 + 2 * valid_pairs  # 1 for '/' + pairs of '1' and '2'
        
        max_length = max(max_length, current_length)

print(max_length)