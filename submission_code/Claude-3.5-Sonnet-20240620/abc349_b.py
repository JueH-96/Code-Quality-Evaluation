# YOUR CODE HERE
def is_good_string(s):
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    frequency_count = {}
    for count in char_count.values():
        frequency_count[count] = frequency_count.get(count, 0) + 1
    
    for freq in frequency_count.values():
        if freq != 0 and freq != 2:
            return False
    
    return True

s = input().strip()
print("Yes" if is_good_string(s) else "No")