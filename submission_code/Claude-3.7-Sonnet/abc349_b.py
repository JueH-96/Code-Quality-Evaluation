def is_good_string(s):
    # Count the frequency of each letter
    letter_freq = {}
    for char in s:
        letter_freq[char] = letter_freq.get(char, 0) + 1
    
    # Count how many letters have each frequency
    freq_count = {}
    for char, freq in letter_freq.items():
        freq_count[freq] = freq_count.get(freq, 0) + 1
    
    # A good string should have exactly 2 letters for each frequency that appears
    for count in freq_count.values():
        if count != 2:
            return False
    
    return True

# Read input
s = input().strip()

# Check if the string is good and print result
if is_good_string(s):
    print("Yes")
else:
    print("No")