def count_candies(N, C, T):
    candies = 0
    last_candy_time = -C  # Initialize to a time before the first possible press

    for t in T:
        if t - last_candy_time >= C:
            candies += 1
            last_candy_time = t  # Update the last candy time to the current press time

    return candies

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    C = int(data[1])
    T = list(map(int, data[2:2 + N]))
    
    result = count_candies(N, C, T)
    print(result)