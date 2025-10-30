import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    names = []
    ages = []
    idx = 1
    for _ in range(N):
        names.append(data[idx]); idx += 1
        ages.append(int(data[idx])); idx += 1

    # Find the index of the youngest person (minimum age)
    start = ages.index(min(ages))

    # Print names starting from the youngest in clockwise order
    for i in range(N):
        print(names[(start + i) % N])

if __name__ == "__main__":
    main()