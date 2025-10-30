N = int(input())

result_string = []
for i in range(1, N + 1):
    if i % 3 == 0:
        result_string.append('x')
    else:
        result_string.append('o')

print("".join(result_string))