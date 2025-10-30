def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # Parse inputs
    N = int(data[0])
    C = int(data[1])
    T = list(map(int, data[2:2+N]))
    
    candies = 0
    last_time = -10**9  # a variable such that first press is always candy
    
    for time in T:
        if time - last_time >= C:
            candies += 1
            last_time = time
            
    print(candies)

if __name__ == '__main__':
    main()