# YOUR CODE HERE
n = int(input())
result = ''
for i in range(n):
    for digit in '02468':
        n -= 1
        if n == 0:
            result += digit
            break
    else:
        continue
    break
print(int(result))