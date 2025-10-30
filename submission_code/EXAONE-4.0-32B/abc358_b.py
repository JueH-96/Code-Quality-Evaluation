import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    a = int(data[1])
    times = list(map(int, data[2:2+n]))
    
    current_end = 0
    results = []
    for t in times:
        start_time = max(t, current_end)
        finish_time = start_time + a
        results.append(str(finish_time))
        current_end = finish_time
    
    print("
".join(results))

if __name__ == "__main__":
    main()