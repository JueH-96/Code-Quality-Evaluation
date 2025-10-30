# YOUR CODE HERE
S = input().strip()

while "WA" in S:
    # Find the leftmost occurrence of "WA"
    pos = S.find("WA")
    # Replace it with "AC"
    S = S[:pos] + "AC" + S[pos+2:]

print(S)