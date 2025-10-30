def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    hands = list(map(int, data[2:]))

    count = 0
    current_stock = M
    for h in hands:
        if current_stock >= h:
            current_stock -= h
            count += 1
        else:
            # Not enough disinfectant for this alien, so it's used up
            current_stock = 0
            break

    print(count)

# Do not forget to call the main function
if __name__ == "__main__":
    main()