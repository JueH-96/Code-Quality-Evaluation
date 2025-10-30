import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    total_sum = sum(a)
    q, r = divmod(total_sum, n)
    a.sort()
    total = 0
    for i in range(n):
        if i < (n - r):
            target = q
        else:
            target = q + 1
        total += abs(a[i] - target)
    print(total // 2)

if __name__ == "__main__":
    main()