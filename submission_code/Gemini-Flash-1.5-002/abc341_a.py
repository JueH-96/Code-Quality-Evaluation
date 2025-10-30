# YOUR CODE HERE
n = int(input())
result = ""
for i in range(n + 1):
    if i % 2 == 0:
        result += "1"
    else:
        result += "0"
print(result)