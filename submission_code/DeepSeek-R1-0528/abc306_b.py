def main():
    data = input().split()
    bits = list(map(int, data))
    total = 0
    for exponent in range(64):
        if bits[exponent] == 1:
            total += (1 << exponent)
    print(total)

if __name__ == "__main__":
    main()