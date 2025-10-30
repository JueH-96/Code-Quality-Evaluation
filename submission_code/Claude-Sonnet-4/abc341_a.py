# YOUR CODE HERE
N = int(input())

result = ""
for i in range(2 * N + 1):
    if i % 2 == 0:  # Even positions get 1
        result += "1"
    else:  # Odd positions get 0
        result += "0"

print(result)