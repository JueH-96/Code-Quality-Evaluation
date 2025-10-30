# YOUR CODE HERE

# Read the inputs
N = int(input())
S = input()
T = input()

# Check if S and T are similar strings
if all(S[i] == T[i] or (S[i] == '1' and T[i] == 'l') or (S[i] == '0' and T[i] == 'o') for i in range(N)):
    print('Yes')
else:
    print('No')