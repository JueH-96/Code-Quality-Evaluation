# Read input
line1 = input().split()
N = int(line1[0])
c_1 = line1[1]
c_2 = line1[2]
S = input()

# Process the string
result = ''.join(char if char == c_1 else c_2 for char in S)

# Print the result
print(result)