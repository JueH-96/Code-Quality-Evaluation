# Read the input
N = int(input())
A = [int(x) for x in input().split()]

# Define a function to check if a string is good
def is_good(S):
    for i in range(N):
        if S[i] == 'A' and S[(i+1)%N] == 'R' and S[(i+2)%N] == 'C':
            if A[i] == 0 and A[(i+1)%N] == 0:
                A[i] = 1
                A[(i+1)%N] = 1
        elif S[(i+2)%N] == 'A' and S[(i+1)%N] == 'R' and S[i] == 'C':
            if A[i] == 0 and A[(i+1)%N] == 0:
                A[i] = 1
                A[(i+1)%N] = 1
    return all(x == 1 for x in A)

# Check if there exists a good string
for i in range(26**N):
    S = ''.join(chr(ord('A') + i//26**j % 26) for j in range(N-1,-1,-1))
    if is_good(S):
        print('Yes')
        break
else:
    print('No')