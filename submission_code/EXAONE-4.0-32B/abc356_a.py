def main():
    import sys
    data = sys.stdin.readline().split()
    n = int(data[0])
    l = int(data[1])
    r = int(data[2])
    arr = list(range(1, n + 1))
    arr[l - 1:r] = arr[l - 1:r][::-1]
    print(' '.join(map(str, arr)))

if __name__ == "__main__":
    main()