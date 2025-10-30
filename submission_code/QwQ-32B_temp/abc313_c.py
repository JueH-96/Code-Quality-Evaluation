import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    total_sum = sum(a)
    k = total_sum // n
    r = total_sum % n
    a_sorted = sorted(a)
    result = 0
    for i in range(n):
        if i < (n - r):
            target = k
        else:
            target = k + 1
        current = a_sorted[i]
        if current > target:
            result += current - target
    print(result)

if __name__ == "__main__":
    main()