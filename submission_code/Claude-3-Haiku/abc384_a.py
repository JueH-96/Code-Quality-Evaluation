# Read the input
N, c1, c2 = input().split()
N = int(N)
S = input()

# Replace characters in S
result = ""
for char in S:
    if char == c1:
        result += c1
    else:
        result += c2

# Print the result
print(result)