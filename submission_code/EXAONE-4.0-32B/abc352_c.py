def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    total_A = 0
    max_diff = 0
    
    for i in range(1, n + 1):
        line = data[i].split()
        a = int(line[0])
        b = int(line[1])
        total_A += a
        diff = b - a
        if diff > max_diff:
            max_diff = diff
            
    result = total_A + max_diff
    print(result)

if __name__ == "__main__":
    main()