import sys

def main():
    n = int(sys.stdin.readline())
    sum_A = 0
    max_diff = 0
    for _ in range(n):
        a, b = map(int, sys.stdin.readline().split())
        sum_A += a
        diff = b - a
        if diff > max_diff:
            max_diff = diff
    print(sum_A + max_diff)

if __name__ == "__main__":
    main()