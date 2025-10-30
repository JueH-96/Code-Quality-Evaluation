# YOUR CODE HERE
N = int(input())
S = list(input())  # Convert to list for easier character modification
Q = int(input())

for _ in range(Q):
    line = input().split()
    t = int(line[0])
    
    if t == 1:
        # Change x-th character to c
        x = int(line[1])
        c = line[2]
        S[x-1] = c  # Convert to 0-based indexing
    elif t == 2:
        # Convert all uppercase to lowercase
        for i in range(len(S)):
            if S[i].isupper():
                S[i] = S[i].lower()
    elif t == 3:
        # Convert all lowercase to uppercase
        for i in range(len(S)):
            if S[i].islower():
                S[i] = S[i].upper()

print(''.join(S))