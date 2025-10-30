# YOUR CODE HERE
N = int(input())
S = input()

# Check for adjacent pairs of 'a' and 'b'
for i in range(N-1):
    if (S[i] == 'a' and S[i+1] == 'b') or (S[i] == 'b' and S[i+1] == 'a'):
        print("Yes")
        break
else:
    print("No")