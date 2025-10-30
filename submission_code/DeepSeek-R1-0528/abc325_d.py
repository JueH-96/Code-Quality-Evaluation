import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    events = []
    index = 1
    for i in range(n):
        t = int(data[index])
        d = int(data[index+1])
        index += 2
        events.append((t, d))
    
    events.sort(key=lambda x: (x[0] + x[1], x[0]))
    
    last = -10**20
    count = 0
    for t, d in events:
        candidate = last + 1
        if candidate < t:
            candidate = t
        if candidate <= t + d:
            count += 1
            last = candidate
            
    print(count)

if __name__ == "__main__":
    main()