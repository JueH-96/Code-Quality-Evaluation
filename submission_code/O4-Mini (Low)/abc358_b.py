import sys
import threading

def main():
    import sys

    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    A = int(input_data[1])
    T = list(map(int, input_data[2:]))

    # current_time tracks when the booth becomes free next
    current_time = 0

    for i in range(N):
        arrival = T[i]
        # If the booth is free before or exactly when this person arrives,
        # they start immediately at arrival.
        # Otherwise, they wait until current_time.
        start = max(arrival, current_time)
        finish = start + A
        print(finish)
        # Update the booth free time
        current_time = finish

if __name__ == "__main__":
    main()