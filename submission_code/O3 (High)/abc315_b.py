import sys

def main() -> None:
    data = list(map(int, sys.stdin.read().split()))
    if not data:
        return
    M = data[0]
    D = data[1:]

    total_days = sum(D)
    middle = (total_days + 1) // 2       # 1-based index of the middle day

    acc = 0                              # days counted so far
    for i, days in enumerate(D):
        if acc + days >= middle:         # the middle day is in month (i+1)
            month = i + 1
            day   = middle - acc
            print(month, day)
            return
        acc += days

if __name__ == "__main__":
    main()