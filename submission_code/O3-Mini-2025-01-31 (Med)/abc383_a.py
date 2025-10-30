def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    water = 0
    prev_time = 0
    index = 1

    for _ in range(N):
        T = int(input_data[index])
        V = int(input_data[index + 1])
        index += 2
        # Water decreases by 1 liter per unit time from prev_time to T
        time_diff = T - prev_time
        water = max(0, water - time_diff)
        # Add water V liters at time T
        water += V
        prev_time = T

    sys.stdout.write(str(water))


if __name__ == '__main__':
    main()