# YOUR CODE HERE
N = int(input())
S = input()

position = S.find("ABC")
if position != -1:
    print(position + 1)
else:
    print(-1)