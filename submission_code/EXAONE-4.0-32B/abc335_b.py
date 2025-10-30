def main():
    N = int(input().strip())
    for x in range(0, N + 1):
        for y in range(0, N - x + 1):
            for z in range(0, N - x - y + 1):
                print(f"{x} {y} {z}")

if __name__ == "__main__":
    main()