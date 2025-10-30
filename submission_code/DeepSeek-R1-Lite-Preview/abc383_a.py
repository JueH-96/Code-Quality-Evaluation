def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    current_time = 0
    current_water = 0
    
    index = 1
    for _ in range(N):
        T = int(data[index])
        V = int(data[index + 1])
        index += 2
        time_diff = T - current_time
        leaked = min(time_diff, current_water)
        current_water -= leaked
        current_water += V
        current_time = T
    
    print(current_water)

if __name__ == '__main__':
    main()