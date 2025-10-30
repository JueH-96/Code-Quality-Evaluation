# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    t = int(data[index])
    index += 1
    results = []
    
    for _ in range(t):
        S = data[index]
        index += 1
        X = data[index]
        index += 1
        Y = data[index]
        index += 1
        
        count_0_X = X.count('0')
        count_0_Y = Y.count('0')
        
        if count_0_X == count_0_Y:
            results.append("Yes")
        else:
            results.append("No")
    
    sys.stdout.write("
".join(results) + "
")