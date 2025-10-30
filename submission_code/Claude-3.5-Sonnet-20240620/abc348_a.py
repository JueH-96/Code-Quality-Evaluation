# YOUR CODE HERE
N = int(input())
result = ''
for i in range(1, N + 1):
    if i % 3 == 0:
        result += 'x'
    else:
        result += 'o'
print(result)