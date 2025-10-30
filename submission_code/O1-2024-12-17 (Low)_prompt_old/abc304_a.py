def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    names = data[1::2]
    ages = list(map(int, data[2::2]))

    # Find the index of youngest person
    min_age = min(ages)
    start_idx = ages.index(min_age)

    # Print names in clockwise order starting from youngest
    for i in range(N):
        print(names[(start_idx + i) % N])

def main():
    solve()

if __name__ == "__main__":
    main()