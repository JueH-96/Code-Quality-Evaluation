def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    a_list = list(map(int, data[2:2+m]))
    
    set_a = set(a_list)
    missing = []
    for num in range(1, n+1):
        if num not in set_a:
            missing.append(num)
            
    print(len(missing))
    if missing:
        print(" ".join(map(str, missing)))

if __name__ == "__main__":
    main()