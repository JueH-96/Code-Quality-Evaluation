n = int(input())
s = input().strip()

# For each character, find the maximum consecutive length
char_max_length = {}

i = 0
while i < n:
    char = s[i]
    length = 1
    
    # Count consecutive occurrences of the same character
    while i + length < n and s[i + length] == char:
        length += 1
    
    # Update the maximum length for this character
    char_max_length[char] = max(char_max_length.get(char, 0), length)
    
    i += length

# Sum up all the maximum lengths
print(sum(char_max_length.values()))