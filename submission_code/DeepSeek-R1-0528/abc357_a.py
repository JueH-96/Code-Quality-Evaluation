def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    h_list = list(map(int, data[2:2+n]))
    
    current = m
    count = 0
    for h in h_list:
        if current >= h:
            current -= h
            count += 1
        else:
            break
    print(count)

if __name__ == "__main__":
    main()