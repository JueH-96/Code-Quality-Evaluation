def is_1122_string(s):
    # Condition 1: Length of the string must be even
    if len(s) % 2 != 0:
        return False
    
    # Condition 2: For each integer i satisfying 1 <= i <= len(s)/2,
    # the (2i-1)-th and 2i-th characters of s should be equal.
    for i in range(0, len(s), 2):
        if s[i] != s[i + 1]:
            return False
    
    # Condition 3: Each character must appear exactly 0 or 2 times in the string.
    char_count = {}
    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    
    for count in char_count.values():
        if count != 2:
            return False
    
    return True

# Read the input string
s = input().strip()
if is_1122_string(s):
    print("Yes")
else:
    print("No")