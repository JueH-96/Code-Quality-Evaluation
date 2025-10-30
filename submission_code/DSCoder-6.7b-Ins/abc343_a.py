# YOUR CODE HERE
A, B = map(int, input().split())
print((set(range(10)) - {A, B}).pop())