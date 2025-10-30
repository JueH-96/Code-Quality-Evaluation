# YOUR CODE HERE
N = int(input().strip())
result = []

for i in range(N + 1):
    result.append('1')
    if i < N:
        result.append('0')

print(''.join(result))