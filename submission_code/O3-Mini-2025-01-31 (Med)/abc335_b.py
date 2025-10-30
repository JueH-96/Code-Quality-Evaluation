def main():
    import sys
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    N = int(input_data)
    
    # Iterate over all possible values for x, y, and z in lexicographical order.
    for x in range(N + 1):
        for y in range(N + 1 - x):  # y can go up to N - x to ensure x+y <= N
            for z in range(N + 1 - x - y):  # similarly, z can go up to N - (x+y)
                print(x, y, z)

if __name__ == '__main__':
    main()