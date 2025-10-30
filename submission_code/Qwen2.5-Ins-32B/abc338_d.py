# YOUR CODE HERE
import sys
input = sys.stdin.readline

def solve():
    N, M = map(int, input().split())
    X = list(map(int, input().split()))
    X = [x-1 for x in X]
    
    # Find the minimum and maximum indices in the tour
    min_idx = min(X)
    max_idx = max(X)
    
    # Calculate the length of the tour if the bridge between min_idx and max_idx is closed
    length1 = (max_idx - min_idx) + (N - (max_idx - min_idx))
    
    # Calculate the length of the tour if the bridge between max_idx and min_idx (in the circular manner) is closed
    length2 = (N - (max_idx - min_idx)) + (max_idx - min_idx)
    
    # Calculate the length of the tour if the bridge between the start and end of the tour is closed
    length3 = (X[-1] - X[0] + N) % N
    
    # The minimum length of the tour is the minimum of the three lengths calculated above
    min_length = min(length1, length2, length3)
    
    # Subtract the number of unique islands visited from the minimum length to get the final answer
    unique_islands = len(set(X))
    print(min_length - unique_islands + 1)

solve()