# YOUR CODE HERE
import sys

def min_passengers(n, changes):
    current_passengers = 0
    min_passengers = 0
    
    for change in changes:
        current_passengers += change
        if current_passengers < 0:
            min_passengers += -current_passengers
            current_passengers = 0
    
    return min_passengers

if __name__ == "__main__":
    n = int(input().strip())
    changes = list(map(int, input().strip().split()))
    print(min_passengers(n, changes))