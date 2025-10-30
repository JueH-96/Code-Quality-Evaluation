import sys

def main() -> None:
    # Read N and D
    N_D_line = sys.stdin.readline().strip()
    while N_D_line == '':
        N_D_line = sys.stdin.readline().strip()
    N_str, D_str = N_D_line.split()
    N, D = int(N_str), int(D_str)

    schedules = [sys.stdin.readline().strip() for _ in range(N)]

    # Determine for each day whether every person is free
    free_days = [all(schedule[day] == 'o' for schedule in schedules) for day in range(D)]

    # Find the longest consecutive run of True values
    max_run = current_run = 0
    for is_free in free_days:
        if is_free:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    print(max_run)

if __name__ == "__main__":
    main()