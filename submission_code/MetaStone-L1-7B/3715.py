import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    k = int(data[1])
    segments = []
    index = 2
    for _ in range(n):
        l = int(data[index])
        r = int(data[index + 1])
        c = int(data[index + 2])
        segments.append((l, r, c))
        index += 3
    
    events = []
    for l, r, c in segments:
        start = l
        end = r + 1
        events.append((start, c))
        events.append((end, -c))
    
    events.sort()
    
    current_sum = 0
    max_sum = 0
    for event in events:
        current_sum += event[1]
        if current_sum > max_sum:
            max_sum = current_sum
    
    print(max_sum)

if __name__ == "__main__":
    main()