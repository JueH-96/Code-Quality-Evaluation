# YOUR CODE HERE
N, c_1, c_2 = input().split()
N = int(N)
S = input()

result = ''.join([c_2 if char != c_1 else char for char in S])
print(result)