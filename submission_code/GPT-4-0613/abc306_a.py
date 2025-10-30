# YOUR CODE HERE
N = int(input().strip())
S = input().strip()

result = ''
for i in range(N):
    result += S[i]*2
print(result)