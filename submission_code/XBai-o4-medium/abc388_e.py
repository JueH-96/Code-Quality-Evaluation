import sys

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    i = 0
    j = 1
    count = 0
    while i < n and j < n:
        if a[j] >= 2 * a[i]:
            count += 1
            i += 1
            j += 1
        else:
            j += 1
    print(count)

if __name__ == '__main__':
    main()