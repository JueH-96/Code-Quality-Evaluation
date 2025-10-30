# YOUR CODE HERE
N = int(input())
result = ''.join(['o' if (i+1) % 3 != 0 else 'x' for i in range(N)])
print(result)