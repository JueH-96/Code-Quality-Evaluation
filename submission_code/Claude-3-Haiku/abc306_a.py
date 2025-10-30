# Read the input
N = int(input())
S = input()

# Concatenate the characters as per the problem statement
result = ""
for i in range(N):
    result += S[i] * 2
print(result)