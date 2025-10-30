import sys

def calculate_success_rate(a, b):
    """Calculate the success rate of a person."""
    return a / (a + b)

def main():
    """Read input, calculate success rates, sort people, and print the result."""
    n = int(sys.stdin.readline().strip())
    people = []
    for i in range(n):
        a, b = map(int, sys.stdin.readline().strip().split())
        success_rate = calculate_success_rate(a, b)
        people.append((success_rate, i + 1))

    # Sort people in descending order of their success rates, with ties broken in ascending order of their assigned numbers
    people.sort(key=lambda x: (-x[0], x[1]))

    # Print the numbers of people in the sorted order
    print(' '.join(map(str, [person[1] for person in people])))

if __name__ == "__main__":
    main()