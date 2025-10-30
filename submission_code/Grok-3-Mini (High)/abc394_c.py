import sys

# Read the input string from stdin
data = sys.stdin.read().strip()
s_list = list(data)
n = len(s_list)

# Perform the replacements until no "WA" is found
while True:
    replaced = False
    for i in range(n - 1):
        if s_list[i] == 'W' and s_list[i + 1] == 'A':
            # Replace "WA" with "AC"
            s_list[i] = 'A'
            s_list[i + 1] = 'C'
            replaced = True
            # Break after replacing the leftmost occurrence
            break
    if not replaced:
        # No replacement was made, exit the loop
        break

# Join the list back into a string and print the result
result = ''.join(s_list)
print(result)