def main():
    N = int(input().strip())
    # Construct the Dragon String: one 'L', N times 'o', then 'n' and 'g'
    dragon_string = "L" + "o" * N + "ng"
    print(dragon_string)

if __name__ == "__main__":
    main()