def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    x = int(data[1])
    scores = list(map(int, data[2:2+n]))
    
    total = 0
    for s in scores:
        if s <= x:
            total += s
            
    print(total)

if __name__ == "__main__":
    main()