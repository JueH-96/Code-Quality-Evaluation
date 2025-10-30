# Read input
N = int(input())
S = [input().strip() for _ in range(N)]

# Check each consecutive pair
for i in range(N - 1):
    if S[i] == 'sweet' and S[i+1] == 'sweet':
        # If the pair is not the last two dishes, output 'No'
        if i + 1 != N - 1:
            print("No")
            exit()

print("Yes")