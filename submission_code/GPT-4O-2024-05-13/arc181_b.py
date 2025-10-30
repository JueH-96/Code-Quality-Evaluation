# YOUR CODE HERE
import sys
input = sys.stdin.read

def solve():
    data = input().split()
    t = int(data[0])
    index = 1
    results = []
    
    for _ in range(t):
        S = data[index]
        X = data[index + 1]
        Y = data[index + 2]
        index += 3
        
        count_X_0 = X.count('0')
        count_X_1 = X.count('1')
        count_Y_0 = Y.count('0')
        count_Y_1 = Y.count('1')
        
        if count_X_0 == count_Y_0 and count_X_1 == count_Y_1:
            results.append("Yes")
        else:
            results.append("No")
    
    sys.stdout.write("
".join(results) + "
")

solve()