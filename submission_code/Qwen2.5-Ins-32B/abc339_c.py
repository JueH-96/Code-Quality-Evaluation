import sys

def min_passengers(N, stops):
    min_passengers = 0
    current_passengers = 0
    for stop in stops:
        current_passengers += stop
        if current_passengers < 0:
            min_passengers += -current_passengers
            current_passengers = 0
    return min_passengers

if __name__ == "__main__":
    N = int(input().strip())
    stops = list(map(int, input().strip().split()))
    print(min_passengers(N, stops))