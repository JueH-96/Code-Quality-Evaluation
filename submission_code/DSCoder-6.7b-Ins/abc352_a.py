# YOUR CODE HERE

def train_stops(N, X, Y, Z):
    if X > Y:
        X, Y = Y, X
    if Z >= X and Z <= Y:
        return "Yes"
    else:
        return "No"

N, X, Y, Z = map(int, input().split())
print(train_stops(N, X, Y, Z))