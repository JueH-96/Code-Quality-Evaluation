import sys

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    Q = int(data[1])
    treatments = list(map(int, data[2:]))

    teeth = [True] * N

    for treatment in treatments:
        teeth[treatment - 1] = not teeth[treatment - 1]

    remaining_teeth = sum(teeth)
    print(remaining_teeth)

if __name__ == "__main__":
    main()