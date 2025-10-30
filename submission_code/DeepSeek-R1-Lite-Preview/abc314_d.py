def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    S = list(data[1])
    Q = int(data[2])
    
    operations = data[3:3+Q]
    
    for op in operations:
        parts = op.split()
        t = int(parts[0])
        if t == 1:
            x = int(parts[1]) - 1  # Convert to 0-based index
            c = parts[2]
            S[x] = c
        elif t == 2:
            S = list(map(str.lower, S))
        elif t == 3:
            S = list(map(str.upper, S))
    
    print(''.join(S))

if __name__ == "__main__":
    main()