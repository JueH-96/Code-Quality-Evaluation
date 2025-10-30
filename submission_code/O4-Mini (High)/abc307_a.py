def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    steps = list(map(int, data[1:]))
    res = []
    for i in range(n):
        week_sum = sum(steps[7*i : 7*(i+1)])
        res.append(str(week_sum))
    print(" ".join(res))

if __name__ == "__main__":
    main()