# YOUR CODE HERE
import sys
input = sys.stdin.read

def minimize_sum_of_coordinates():
    data = input().strip().split()
    N = int(data[0])
    X = list(map(int, data[1:]))
    
    # Since X is already sorted, we can directly find the median
    if N % 2 == 0:
        median = (X[N//2 - 1] + X[N//2]) / 2
    else:
        median = X[N//2]
    
    # Calculate the minimum possible sum
    min_sum = sum(X)
    
    print(int(min_sum))

minimize_sum_of_coordinates()