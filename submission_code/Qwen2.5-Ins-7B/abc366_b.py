# YOUR CODE HERE
n = int(input())
strings = [input() for _ in range(n)]
max_len = max(len(s) for s in strings)

result = [''] * max_len
for i in range(max_len):
    for j in range(n):
        if i < len(strings[j]):
            result[i] += strings[j][i]
        else:
            result[i] += '*'

for r in result:
    print(r.rstrip('*'))