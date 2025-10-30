import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))

    # Create a boolean array marking firework days
    has_fire = [False] * (N + 2)
    for a in A:
        has_fire[a] = True

    # next_fire[i] will hold the next day >= i with fireworks
    next_fire = [0] * (N + 2)
    # Initialize beyond the festival to a large sentinel
    next_fire[N + 1] = 10**9

    # Build next_fire backwards
    for day in range(N, 0, -1):
        if has_fire[day]:
            next_fire[day] = day
        else:
            next_fire[day] = next_fire[day + 1]

    # For each day i, answer is next_fire[i] - i
    out = []
    for i in range(1, N + 1):
        out.append(str(next_fire[i] - i))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()