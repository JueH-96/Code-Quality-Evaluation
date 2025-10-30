import sys

def main():
    n = int(sys.stdin.readline())
    bases = []
    for _ in range(n):
        w, x = map(int, sys.stdin.readline().split())
        bases.append((w, x))
    
    max_total = 0
    for T in range(24):
        current = 0
        for w, x in bases:
            local_start = (T + x) % 24
            if 9 <= local_start <= 17:
                current += w
        if current > max_total:
            max_total = current
    print(max_total)

if __name__ == "__main__":
    main()