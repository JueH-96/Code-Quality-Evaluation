def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, A = map(int, data[:2])
    T = list(map(int, data[2:]))

    finish_times = []
    free_time = 0  # The earliest time when the booth becomes free.

    for arr_time in T:
        if arr_time >= free_time:
            # Booth is free when they arrive
            finish = arr_time + A
        else:
            # Booth is still busy, they start immediately after it becomes free
            finish = free_time + A
        finish_times.append(finish)
        free_time = finish  # Update the next free time

    print("
".join(map(str, finish_times)))

def main():
    solve()

if __name__ == "__main__":
    main()