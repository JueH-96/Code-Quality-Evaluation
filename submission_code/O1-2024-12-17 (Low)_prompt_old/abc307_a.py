def solve():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    steps = list(map(int, data[1:]))

    # Each week has 7 days, so we group steps in blocks of 7
    results = []
    for i in range(N):
        week_sum = sum(steps[i*7:(i+1)*7])
        results.append(str(week_sum))

    print(" ".join(results))

def main():
    solve()

if __name__ == "__main__":
    main()