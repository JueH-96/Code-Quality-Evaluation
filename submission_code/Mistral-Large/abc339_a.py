import sys

input = sys.stdin.read
data = input().strip()

# Split the string by '.' and take the last element
result = data.split('.')[-1]

# Print the result
print(result)