def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    pairs = list(zip(map(int, data[1::2]), map(int, data[2::2])))
    water = 0
    previous_time = 0
    for T, V in pairs:
        time_difference = T - previous_time
        water = max(0, water - time_difference)
        water += V
        previous_time = T
    print(water)

if __name__ == "__main__":
    main()