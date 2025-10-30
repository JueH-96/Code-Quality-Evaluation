import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    Q = int(data[0])
    plants = []
    start = 0
    current_days = 0
    
    for i in range(1, Q + 1):
        parts = data[i].split()
        if parts[0] == '1':
            plants.append(current_days)
        elif parts[0] == '2':
            T = int(parts[1])
            current_days += T
        elif parts[0] == '3':
            H = int(parts[1])
            threshold = current_days - H
            new_start = bisect.bisect_right(plants, threshold, start, len(plants))
            count = new_start - start
            print(count)
            start = new_start

if __name__ == "__main__":
    main()