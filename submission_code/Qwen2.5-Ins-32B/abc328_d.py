S = input().strip()

i = 0
while i < len(S) - 2:
    if S[i:i+3] == "ABC":
        S = S[:i] + S[i+3:]
        i = max(0, i-2)  # Move back 2 steps to check for new "ABC" after removal
    else:
        i += 1

print(S)