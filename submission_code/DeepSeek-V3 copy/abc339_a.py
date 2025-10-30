# Read the input string
S = input().strip()
# Split the string by '.' and get the last part
last_substring = S.split('.')[-1]
# Print the result
print(last_substring)