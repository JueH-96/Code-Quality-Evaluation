def main():
    N = int(input().strip())
    S = input().strip()

    # Initialize a set to keep track of unique characters seen
    seen_chars = set()

    # Iterate over the string and add characters to the set
    for i, char in enumerate(S, 1):
        seen_chars.add(char)
        # Once all three characters have been seen, break the loop
        if len(seen_chars) == 3:
            print(i)
            break

if __name__ == "__main__":
    main()