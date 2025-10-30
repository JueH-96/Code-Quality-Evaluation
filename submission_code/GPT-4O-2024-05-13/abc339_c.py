# YOUR CODE HERE
import sys
input = sys.stdin.read

def min_passengers(N, A):
    min_passengers = 0
    current_passengers = 0
    
    for change in A:
        current_passengers += change
        if current_passengers < 0:
            min_passengers += abs(current_passengers)
            current_passengers = 0
    
    return min_passengers

data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

print(min_passengers(N, A))