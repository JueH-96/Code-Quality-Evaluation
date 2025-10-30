# YOUR CODE HERE
N = int(input())
S = input().strip()

result = ''.join(char * 2 for char in S)
print(result)