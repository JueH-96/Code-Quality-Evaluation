def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    S = int(data[1])
    K = int(data[2])
    
    index = 3
    total = 0
    for _ in range(n):
        p = int(data[index])
        q = int(data[index + 1])
        index += 2
        total += p * q
    
    if total < S:
        total += K
    
    print(total)

if __name__ == "__main__":
    main()