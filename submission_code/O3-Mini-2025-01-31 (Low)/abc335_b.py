def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    
    for x in range(N + 1):
        for y in range(N + 1):
            for z in range(N + 1):
                if x + y + z <= N:
                    print(x, y, z)

if __name__ == '__main__':
    main()