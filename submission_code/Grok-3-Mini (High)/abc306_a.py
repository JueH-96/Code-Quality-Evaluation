# Read the integer N from input
N = int(input())
# Read the string S from input
S = input()
# Build the new string by repeating each character twice and join them
result = ''.join(c * 2 for c in S)
# Print the result
print(result)