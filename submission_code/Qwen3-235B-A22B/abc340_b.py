def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    Q = int(data[0])
    A = []
    res = []
    
    for line in data[1:]:
        parts = line.split()
        if parts[0] == '1':
            x = int(parts[1])
            A.append(x)
        else:
            k = int(parts[1])
            res.append(str(A[-k]))
    
    print('
'.join(res))

if __name__ == "__main__":
    main()