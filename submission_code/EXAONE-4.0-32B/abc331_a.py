def main():
    import sys
    data = sys.stdin.read().split()
    M = int(data[0])
    D = int(data[1])
    y = int(data[2])
    m = int(data[3])
    d = int(data[4])
    
    if d < D:
        print(f"{y} {m} {d+1}")
    else:
        if m < M:
            print(f"{y} {m+1} 1")
        else:
            print(f"{y+1} 1 1")

if __name__ == "__main__":
    main()