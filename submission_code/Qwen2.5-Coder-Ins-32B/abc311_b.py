# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    D = int(input[1])
    schedules = input[2:]

    max_free_days = 0
    current_free_days = 0

    for day in range(D):
        if all(schedules[i][day] == 'o' for i in range(N)):
            current_free_days += 1
            max_free_days = max(max_free_days, current_free_days)
        else:
            current_free_days = 0

    print(max_free_days)

if __name__ == "__main__":
    main()