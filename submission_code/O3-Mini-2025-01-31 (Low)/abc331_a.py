def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    M = int(input_data[0])
    D = int(input_data[1])
    y = int(input_data[2])
    m = int(input_data[3])
    d = int(input_data[4])
    
    # Compute next day
    if d < D:
        d += 1
    else:
        d = 1
        if m < M:
            m += 1
        else:
            m = 1
            y += 1

    sys.stdout.write(f"{y} {m} {d}")

if __name__ == '__main__':
    main()