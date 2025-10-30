import sys

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    current_passengers = 0
    min_passengers = 0

    for i in range(N):
        current_passengers += A[i]
        if current_passengers < 0:
            min_passengers += -current_passengers
            current_passengers = 0

    print(current_passengers + min_passengers)

if __name__ == "__main__":
    main()