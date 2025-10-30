def main():
    from collections import Counter

    A1, A2, A3, A4 = map(int, input().split())
    counts = Counter([A1, A2, A3, A4])

    # The maximum number of times we can discard pairs
    # is the sum of (count_of_color // 2) over all colors.
    answer = sum(count // 2 for count in counts.values())

    print(answer)

# Do not remove
if __name__ == "__main__":
    main()