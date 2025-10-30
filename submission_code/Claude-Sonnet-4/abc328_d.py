# YOUR CODE HERE
S = input().strip()

while True:
    pos = S.find("ABC")
    if pos == -1:
        break
    S = S[:pos] + S[pos+3:]

print(S)