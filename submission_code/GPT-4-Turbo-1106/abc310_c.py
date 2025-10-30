def main():
    N = int(input().strip())
    sticks = set()

    for _ in range(N):
        S = input().strip()
        # Add both the string and its reverse to the set to handle the "same" condition
        sticks.add(tuple(sorted([S, S[::-1]])))

    print(len(sticks))

if __name__ == "__main__":
    main()