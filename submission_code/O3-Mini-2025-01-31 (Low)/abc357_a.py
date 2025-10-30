def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    # Parse inputs
    N = int(input_data[0])
    M = int(input_data[1])
    H = list(map(int, input_data[2:]))
    
    count = 0
    remaining = M
    for h in H:
        if remaining >= h:
            remaining -= h
            count += 1
        else:
            break
    
    print(count)

if __name__ == "__main__":
    main()