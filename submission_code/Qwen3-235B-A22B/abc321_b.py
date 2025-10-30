import sys

def main():
    n, x = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    sum_a = sum(a)
    min_a = min(a)
    max_a = max(a)
    
    for y in range(101):
        new_min = min(min_a, y)
        new_max = max(max_a, y)
        current = sum_a + y - new_min - new_max
        if current >= x:
            print(y)
            return
    print(-1)

if __name__ == "__main__":
    main()