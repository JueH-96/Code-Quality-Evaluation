# YOUR CODE HERE
A, B = map(int, input().split())
result = [i for i in range(10) if i != A + B][0]
print(result)