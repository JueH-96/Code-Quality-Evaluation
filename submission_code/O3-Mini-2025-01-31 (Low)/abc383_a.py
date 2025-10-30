def main():
    import sys
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    water = 0
    prev_time = 0
    index = 1
    for i in range(n):
        t = int(input_data[index])
        v = int(input_data[index+1])
        index += 2
        # Deduct leaked water between prev_time and current time t
        leaked = t - prev_time
        water = max(0, water - leaked)
        water += v
        prev_time = t
    print(water)

if __name__ == '__main__':
    main()