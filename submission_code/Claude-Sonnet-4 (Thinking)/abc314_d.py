N = int(input())
S = input()
Q = int(input())

# Convert string to list for easier modification
S = list(S)

for _ in range(Q):
    line = input().split()
    t = int(line[0])
    
    if t == 1:
        x = int(line[1])
        c = line[2]
        S[x-1] = c  # Convert to 0-indexed
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

# Convert back to string and print
print(''.join(S))