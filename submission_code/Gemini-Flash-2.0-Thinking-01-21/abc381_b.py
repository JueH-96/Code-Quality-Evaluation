import collections

def is_1122_string(s):
    # Condition 1: length must be even
    if len(s) % 2 != 0:
        return False
    # Condition 2: pairs of characters must be equal
    for i in range(len(s) // 2):
        if s[2*i] != s[2*i+1]:
            return False
    # Condition 3: each character appears exactly twice
    char_counts = collections.Counter(s)
    for count in char_counts.values():
        if count != 2:
            return False
    # If all conditions are met, it's a 1122 string
    return True

# Read input and print output
input_string = input()
if is_1122_string(input_string):
    print("Yes")
else:
    print("No")