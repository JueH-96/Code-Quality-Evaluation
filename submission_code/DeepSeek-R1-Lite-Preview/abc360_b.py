S, T = input().split()
len_S = len(S)

found = False

for w in range(1, len_S):
    for c in range(1, w + 1):
        # Split S into chunks of size w
        chunks = [S[i * w : (i + 1) * w] for i in range((len_S + w - 1) // w)]
        # Collect the c-th characters from chunks that have at least c characters
        characters = [chunk[c - 1] for chunk in chunks if len(chunk) >= c]
        res = ''.join(characters)
        if res == T:
            found = True
            break
    if found:
        break

print("Yes" if found else "No")