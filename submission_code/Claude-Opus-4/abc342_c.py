# YOUR CODE HERE
N = int(input())
S = input()
Q = int(input())

# Convert string to list for easier manipulation
s_list = list(S)

for _ in range(Q):
    c, d = input().split()
    # Replace all occurrences of c with d
    for i in range(N):
        if s_list[i] == c:
            s_list[i] = d

# Convert back to string and print
print(''.join(s_list))