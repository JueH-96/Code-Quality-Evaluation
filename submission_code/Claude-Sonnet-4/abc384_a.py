# YOUR CODE HERE
# Read the first line and parse N, c_1, c_2
line1 = input().split()
N = int(line1[0])
c_1 = line1[1]
c_2 = line1[2]

# Read the string S
S = input()

# Replace characters that are not c_1 with c_2
result = ""
for char in S:
    if char != c_1:
        result += c_2
    else:
        result += char

print(result)