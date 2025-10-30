def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = list(map(int, data[1:1+n]))
    
    ptr = 0
    total = 0
    for j in range(1, n):
        while ptr < j and 2 * arr[ptr] <= arr[j]:
            ptr += 1
        total += ptr
        
    print(total)

if __name__ == "__main__":
    main()