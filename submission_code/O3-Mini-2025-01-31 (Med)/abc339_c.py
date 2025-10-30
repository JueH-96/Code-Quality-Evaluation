def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    cum_sum = 0
    min_cum = 0
    for a in A:
        cum_sum += a
        min_cum = min(min_cum, cum_sum)
    
    # The initial number of passengers must be at least -min_cum to keep non-negative counts.
    initial_min = -min_cum if min_cum < 0 else 0
    current_passengers = initial_min + cum_sum
    
    print(current_passengers)

if __name__ == '__main__':
    main()