import sys

def main():
    input = sys.stdin.readline
    N, D = map(int, input().split())
    schedules = [input().strip() for _ in range(N)]
    
    # Determine for each day if everyone is free
    free_days = [True] * D
    for s in schedules:
        for j, ch in enumerate(s):
            if ch == 'x':
                free_days[j] = False
                
    # Find the longest run of consecutive True in free_days
    max_len = 0
    current = 0
    for day_free in free_days:
        if day_free:
            current += 1
            if current > max_len:
                max_len = current
        else:
            current = 0
    
    print(max_len)

if __name__ == "__main__":
    main()