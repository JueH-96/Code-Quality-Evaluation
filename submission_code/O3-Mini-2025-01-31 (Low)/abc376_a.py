def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    C = int(input_data[1])
    T = list(map(int, input_data[2:2+N]))
    
    candy_count = 0
    last_time = -10**9  # initial very old time
    
    for t in T:
        if t - last_time >= C:
            candy_count += 1
            last_time = t
            
    print(candy_count)

if __name__ == '__main__':
    main()