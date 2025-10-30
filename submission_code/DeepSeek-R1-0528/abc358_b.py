def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = int(data[1])
    arrival_times = list(map(int, data[2:2+n]))
    
    current_end = 0
    results = []
    
    for t in arrival_times:
        start_time = max(current_end, t)
        finish_time = start_time + A
        current_end = finish_time
        results.append(finish_time)
        
    for res in results:
        print(res)

if __name__ == "__main__":
    main()