# YOUR CODE HERE
import sys

def min_tour_length(N, M, X):
    def distance(a, b):
        return min(abs(a - b), N - abs(a - b))
    
    def tour_length(X, closed_bridge):
        total_length = 0
        for i in range(M - 1):
            if (X[i] == closed_bridge and X[i + 1] == (closed_bridge % N + 1)) or \
               (X[i] == (closed_bridge % N + 1) and X[i + 1] == closed_bridge):
                total_length += N - 1
            else:
                total_length += distance(X[i], X[i + 1])
        return total_length
    
    min_length = float('inf')
    for bridge in range(1, N + 1):
        min_length = min(min_length, tour_length(X, bridge))
    
    return min_length

input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
X = list(map(int, data[2:]))

print(min_tour_length(N, M, X))