def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    arr = list(map(int, data[2:2+n]))
    
    starts = 0
    current = k
    i = 0
    while i < n:
        if current < arr[i]:
            starts += 1
            current = k
        else:
            current -= arr[i]
            i += 1
    print(starts + 1)

if __name__ == "__main__":
    main()