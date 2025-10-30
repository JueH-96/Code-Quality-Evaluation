def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    M = int(input_data[1])
    A = set(int(x) for x in input_data[2:2+M])
    
    missing = [str(i) for i in range(1, N+1) if i not in A]
    print(len(missing))
    if missing:
        print(" ".join(missing))

if __name__ == '__main__':
    main()