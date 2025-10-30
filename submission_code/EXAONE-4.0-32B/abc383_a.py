def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    events = []
    for i in range(1, n + 1):
        parts = data[i].split()
        t = int(parts[0])
        v = int(parts[1])
        events.append((t, v))
    
    current_water = 0
    last_time = 0
    
    for t, v in events:
        leak_duration = t - last_time
        current_water = max(0, current_water - leak_duration)
        current_water += v
        last_time = t
        
    print(current_water)

if __name__ == "__main__":
    main()