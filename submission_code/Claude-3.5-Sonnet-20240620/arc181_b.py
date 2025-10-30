# YOUR CODE HERE
def can_exist(S, X, Y):
    if len(X) != len(Y):
        return False
    
    for i in range(len(X)):
        if X[i] == Y[i]:
            continue
        if X[i] == '0':
            if S * (i + 1) not in Y[:i+1]:
                return False
        else:
            if S * (i + 1) not in X[:i+1]:
                return False
    
    return True

t = int(input())
for _ in range(t):
    S = input().strip()
    X = input().strip()
    Y = input().strip()
    
    if can_exist(S, X, Y):
        print("Yes")
    else:
        print("No")