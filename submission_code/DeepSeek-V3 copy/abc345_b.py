# YOUR CODE HERE
X = int(input())
if X % 10 == 0:
    print(X // 10)
else:
    if X > 0:
        print((X // 10) + 1)
    else:
        print(X // 10)