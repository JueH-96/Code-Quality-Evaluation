import sys

def main():
    n, d = map(int, sys.stdin.readline().split())
    schedules = [sys.stdin.readline().strip() for _ in range(n)]
    
    all_free = []
    for j in range(d):
        ok = True
        for i in range(n):
            if schedules[i][j] != 'o':
                ok = False
                break
        all_free.append(ok)
    
    max_consec = 0
    current = 0
    for day in all_free:
        if day:
            current += 1
            if current > max_consec:
                max_consec = current
        else:
            current = 0
    print(max_consec)

if __name__ == "__main__":
    main()