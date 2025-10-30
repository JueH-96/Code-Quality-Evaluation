def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    arr = list(map(int, data[1:1+n]))
    
    left = 0
    right = n // 2
    count = 0
    
    while left < n // 2 and right < n:
        if arr[left] * 2 <= arr[right]:
            count += 1
            left += 1
            right += 1
        else:
            right += 1
            
    print(count)

if __name__ == '__main__':
    main()