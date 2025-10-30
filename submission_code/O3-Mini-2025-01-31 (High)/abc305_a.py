def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if input_data:
        N = int(input_data[0])
        # The water station positions are at every 5 km from 0 to 100.
        # Using round() here works because N is an integer, so N/5 cannot be a half-integer.
        nearest_station = 5 * round(N / 5)
        sys.stdout.write(str(nearest_station))

if __name__ == '__main__':
    main()