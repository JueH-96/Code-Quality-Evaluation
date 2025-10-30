# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
results = []

for i in range(t):
    S = data[3*i + 1]
    X = data[3*i + 2]
    Y = data[3*i + 3]
    
    len_X = len(X)
    len_Y = len(Y)
    
    if len_X != len_Y:
        results.append("No")
        continue
    
    if X == Y:
        results.append("Yes")
        continue
    
    # Check if there exists a T such that f(S, T, X) == f(S, T, Y)
    # This is true if and only if the frequency of '0' in X is equal to the frequency of '0' in Y
    count_0_X = X.count('0')
    count_0_Y = Y.count('0')
    
    if count_0_X == count_0_Y:
        results.append("Yes")
    else:
        results.append("No")

# Print all results
for result in results:
    print(result)