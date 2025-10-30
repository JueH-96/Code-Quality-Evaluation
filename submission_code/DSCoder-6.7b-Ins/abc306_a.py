# YOUR CODE HERE

N = int(input())
S = input()

result = ''
for i in range(N):
    result += S[i]*2
result += S[-1]

print(result)