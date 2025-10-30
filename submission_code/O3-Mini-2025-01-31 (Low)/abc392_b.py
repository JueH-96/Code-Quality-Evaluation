def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:2+M]))
    
    present = set(A)
    missing = [num for num in range(1, N+1) if num not in present]
    
    print(len(missing))
    if missing:
        print(" ".join(map(str, missing)))

if __name__ == '__main__':
    main()