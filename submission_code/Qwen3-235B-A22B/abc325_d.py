import sys

def main():
    n = int(sys.stdin.readline())
    products = []
    for _ in range(n):
        t, d = map(int, sys.stdin.readline().split())
        end = t + d
        products.append((end, t))
    # Sort by end time ascending, then T_i ascending
    products.sort(key=lambda x: (x[0], x[1]))
    count = 0
    prev_time = -float('inf')
    for end, t in products:
        candidate = max(prev_time + 1, t)
        if candidate <= end:
            count += 1
            prev_time = candidate
    print(count)

if __name__ == "__main__":
    main()