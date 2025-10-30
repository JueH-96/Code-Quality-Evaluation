# YOUR CODE HERE
a = list(map(int, input().split()))
b = list(map(int, input().split()))
tak = sum(a)
aok = sum(b)
print(max(1, (tak - aok) + 1))