import sys

def find_unique_char_position(s):
    # Count occurrences of each character
    count = {}
    for char in s:
        if char in count:
            count[char] += 1
        else:
            count[char] = 1

    # Find the character that occurs only once
    unique_char = None
    for char, freq in count.items():
        if freq == 1:
            unique_char = char
            break

    # Find the position of the unique character
    for index, char in enumerate(s):
        if char == unique_char:
            return index + 1  # 1-based index

# Read input from stdin
input_data = sys.stdin.read().strip()

# Find and print the position of the unique character
result = find_unique_char_position(input_data)
print(result)