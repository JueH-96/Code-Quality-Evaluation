def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, C = map(int, data[:2])
    T = list(map(int, data[2:]))

    candy_count = 1  # Always receive a candy on the first press
    last_candy_time = T[0]

    for i in range(1, N):
        if T[i] - last_candy_time >= C:
            candy_count += 1
            last_candy_time = T[i]

    print(candy_count)

# Do not forget to call main
main()