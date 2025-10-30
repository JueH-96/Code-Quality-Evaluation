def main():
    import sys
    data = sys.stdin.read().strip()
    X = int(data)

    total = 0
    for i in range(1, 10):
        for j in range(1, 10):
            prod = i * j
            if prod != X:
                total += prod

    print(total)

if __name__ == "__main__":
    main()