# Read the input
N = int(input())
S = input()

# Find the first occurrence of 'ABC' in the string
for i in range(N-2):
    if S[i:i+3] == 'ABC':
        print(i+1)
        break
else:
    print(-1)