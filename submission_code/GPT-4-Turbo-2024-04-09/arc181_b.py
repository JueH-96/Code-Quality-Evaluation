import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    t = int(data[index])
    index += 1
    results = []
    
    for _ in range(t):
        S = data[index]
        X = data[index + 1]
        Y = data[index + 2]
        index += 3
        
        # Count the number of '0's and '1's in X and Y
        count_X0 = X.count('0')
        count_X1 = X.count('1')
        count_Y0 = Y.count('0')
        count_Y1 = Y.count('1')
        
        # If the number of S's required (0's) are different, it's impossible
        if count_X0 != count_Y0:
            results.append("No")
            continue
        
        # If the number of T's required (1's) are different, it's impossible
        if count_X1 != count_Y1:
            results.append("No")
            continue
        
        # If both counts match, it's possible
        results.append("Yes")
    
    # Print all results
    sys.stdout.write("
".join(results) + "
")