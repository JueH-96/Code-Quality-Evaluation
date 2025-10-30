from collections import defaultdict

# KMP algorithm: Get lps array for the given pattern
def compute_lps_array(pattern):
    lps = [0] * len(pattern)
    length = 0  # length of the previous longest prefix suffix
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

n = int(input().strip())
strings = []
for _ in range(n):
    strings.append(input().strip())

unique_sticks = set()

# For each string, consider it and its reverse as potential 'sticks'
for string in strings:
    pattern = string + "#" + string[::-1]
    # Use KMP to find the longest prefix which is also suffix
    lps = compute_lps_array(pattern)
    # The length at the end will tell us how much to remove from total length to get unique part
    unique_part_length = len(pattern) - lps[-1] - 1  # -1 for the '#' character
    # Check if it's unique or a repetition of a substring
    if unique_part_length >= len(string):
        # Building a key that represents the stick along with its reverse
        key = (string, string[::-1])
    else:
        # If it's a repetition, the unique part will be shorter than the string, find the repetition unit
        repetition_unit_length = len(string) // (len(string) - unique_part_length)
        key = (string[:repetition_unit_length], string[-repetition_unit_length:][::-1])
    unique_sticks.add(key)

print(len(unique_sticks))