# YOUR CODE HERE
S = input().strip()

while "WA" in S:
    # Find the index of the leftmost occurrence of "WA"
    index = S.find("WA")
    # Replace the leftmost "WA" with "AC"
    S = S[:index] + "AC" + S[index+2:]

print(S)