def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    c = int(data[1])
    times = list(map(int, data[2:2+n]))
    
    candies = 0
    last_time = None
    
    for t in times:
        if last_time is None:
            candies += 1
            last_time = t
        else:
            if t - last_time >= c:
                candies += 1
                last_time = t
                
    print(candies)

if __name__ == "__main__":
    main()