import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    A = int(data[1])
    T = list(map(int, data[2:2+N]))

    results = []
    current_end = 0  # time when the person currently at the booth will finish

    for arrival in T:
        if arrival >= current_end:
            # Booth is idle when this person arrives
            finish = arrival + A
        else:
            # Person has to wait until current_end
            finish = current_end + A
        current_end = finish
        results.append(finish)

    sys.stdout.write('
'.join(map(str, results)))

if __name__ == "__main__":
    main()