# YOUR CODE HERE
S = input().strip()

while "ABC" in S:
    index = S.find("ABC")
    S = S[:index] + S[index+3:]

print(S)