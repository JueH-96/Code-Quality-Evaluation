import sys

def main():
    N = int(sys.stdin.readline())
    products = []
    for _ in range(N):
        T, D = map(int, sys.stdin.readline().split())
        end = T + D
        products.append((end, T))
    
    # Sort by end time, then by start time (T)
    products.sort(key=lambda x: (x[0], x[1]))
    
    last_time = -float('inf')
    count = 0
    
    for p in products:
        end, T = p
        current_start = T
        current_end = end
        earliest = max(last_time + 1, current_start)
        if earliest <= current_end:
            count += 1
            last_time = earliest
    
    print(count)

if __name__ == "__main__":
    main()