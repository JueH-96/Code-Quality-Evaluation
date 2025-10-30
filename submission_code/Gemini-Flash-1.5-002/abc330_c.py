# YOUR CODE HERE
D = int(input())
ans = float('inf')
for x in range(int((D**0.5)) + 2):
    y_square = D - x * x
    if y_square < 0:
        continue
    y = int(y_square**0.5)
    ans = min(ans, abs(x * x + y * y - D))
    y += 1
    ans = min(ans, abs(x * x + y * y - D))

print(ans)