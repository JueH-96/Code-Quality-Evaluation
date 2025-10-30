import sys

def count_remaining_sets(N, X, Y):
    # Create a list of (x, y) pairs
    balls = [(x, y) for x, y in zip(X, Y)]
    
    # Sort the balls based on the condition
    balls.sort(key=lambda x: (x[0], -x[1]))
    
    # Initialize the count of remaining sets
    count = 0
    
    # Iterate through the sorted balls
    for i in range(N):
        # Check if the current ball is still remaining
        if i == 0 or (balls[i][0] > balls[i-1][0] and balls[i][1] > balls[i-1][1]):
            count += 1
    
    return count % 998244353

# Read input from stdin
N = int(sys.stdin.readline().strip())
X = [int(x) for x in sys.stdin.readline().strip().split()]
Y = [int(y) for y in sys.stdin.readline().strip().split()]

# Solve the problem and write the answer to stdout
print(count_remaining_sets(N, X, Y))