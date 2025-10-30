# YOUR CODE HERE
N = int(input())

result_chars = []
for i in range(1, N + 1):
    if i % 3 == 0:
        result_chars.append('x')
    else:
        result_chars.append('o')

output_string = "".join(result_chars)
print(output_string)