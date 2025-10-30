def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    k = int(data[1])
    arr = list(map(int, data[2:2+n]))
    
    part1 = arr[:n - k]
    part2 = arr[n - k:]
    
    result = part2 + part1
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()