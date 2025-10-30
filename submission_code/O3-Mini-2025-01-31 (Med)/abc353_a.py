def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    buildings = list(map(int, input_data[1:]))
    
    first_height = buildings[0]
    result = -1
    for i in range(1, N):
        if buildings[i] > first_height:
            result = i + 1  # Convert from 0-indexed to 1-indexed position.
            break
    print(result)

if __name__ == '__main__':
    main()