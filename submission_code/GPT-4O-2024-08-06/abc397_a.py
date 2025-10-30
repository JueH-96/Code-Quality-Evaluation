# YOUR CODE HERE
def classify_temperature():
    import sys
    input = sys.stdin.read
    X = float(input().strip())
    
    if X >= 38.0:
        print(1)
    elif X >= 37.5:
        print(2)
    else:
        print(3)