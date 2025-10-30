# YOUR CODE HERE
S = input().strip()
n = len(S)
max_length = 0

for i in range(n):
    for j in range(i, n):
        substring = S[i:j+1]
        if substring == substring[::-1]:  # Check if palindrome
            max_length = max(max_length, j - i + 1)

print(max_length)