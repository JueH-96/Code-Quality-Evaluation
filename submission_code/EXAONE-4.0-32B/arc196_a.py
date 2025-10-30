import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = list(map(int, data[1:1+n]))
    arr.sort()
    k = n // 2
    total = sum(arr[-k:]) - sum(arr[:k])
    print(total)

if __name__ == '__main__':
    main()