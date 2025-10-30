# YOUR CODE HERE
S, T = input().split()
n = len(S)

for w in range(1, n):
    for c in range(1, w + 1):
        # Extract c-th character from chunks of size w where chunk length >= c
        extracted = ""
        for i in range(0, n, w):
            chunk = S[i:i+w]
            if len(chunk) >= c:
                extracted += chunk[c-1]  # c is 1-indexed
        
        # Check if the concatenation equals T
        if extracted == T:
            print("Yes")
            exit()

print("No")