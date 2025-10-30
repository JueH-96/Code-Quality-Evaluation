# YOUR CODE HERE
S = input().strip()
first = S.find('|')
second = S.find('|', first + 1)
print(S[:first] + S[second + 1:])