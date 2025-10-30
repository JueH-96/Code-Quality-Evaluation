import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    if n == 1:
        print(1)
        return
    diff = [a[i] - a[i-1] for i in range(1, n)]
    prev = diff[0]
    current_run = 1
    total = 0
    for i in range(1, len(diff)):
        if diff[i] == prev:
            current_run += 1
        else:
            total += current_run * (current_run + 1) // 2
            current_run = 1
            prev = diff[i]
    # Add the last run
    total += current_run * (current_run + 1) // 2
    print(total + n)

if __name__ == "__main__":
    main()