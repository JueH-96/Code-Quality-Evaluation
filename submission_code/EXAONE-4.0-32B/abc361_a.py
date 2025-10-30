def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    k = int(data[1])
    x = int(data[2])
    A = list(map(int, data[3:3+n]))
    
    B = A[:k] + [x] + A[k:]
    
    print(' '.join(map(str, B)))

if __name__ == '__main__':
    main()