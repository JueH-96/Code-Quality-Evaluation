import sys

def main():
    N = int(input().strip())
    A = list(map(int, input().strip().split()))

    # Initialize the minimum number of passengers at any point to 0
    min_passengers = 0
    current_passengers = 0

    # Iterate over the stops
    for change in A:
        current_passengers += change
        # Update the minimum number of passengers if the current number is less than the minimum
        min_passengers = min(min_passengers, current_passengers)

    # The minimum possible current number of passengers is the absolute value of min_passengers
    print(abs(min_passengers))

if __name__ == "__main__":
    main()