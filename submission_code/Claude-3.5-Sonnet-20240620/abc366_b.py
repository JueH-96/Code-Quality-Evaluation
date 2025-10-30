# YOUR CODE HERE
N = int(input())
strings = [input().strip() for _ in range(N)]

max_length = max(len(s) for s in strings)
result = [['*'] * N for _ in range(max_length)]

for i, s in enumerate(strings):
    for j, char in enumerate(s):
        result[j][N - i - 1] = char

for row in result:
    while row[-1] == '*':
        row.pop()
    print(''.join(row))