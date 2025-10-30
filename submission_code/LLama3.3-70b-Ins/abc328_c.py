def count_consecutive_pairs(s, l, r):
    """
    Count the number of places where the same lowercase English letter occurs twice in a row
    in the substring s[l-1:r] of s.

    Args:
        s (str): The input string.
        l (int): The start index of the substring (1-indexed).
        r (int): The end index of the substring (1-indexed).

    Returns:
        int: The number of places where the same lowercase English letter occurs twice in a row.
    """
    count = 0
    for i in range(l - 1, r - 1):
        if s[i] == s[i + 1]:
            count += 1
    return count


def main():
    # Read the input
    n, q = map(int, input().split())
    s = input()
    queries = []
    for _ in range(q):
        l, r = map(int, input().split())
        queries.append((l, r))

    # Process the queries
    for l, r in queries:
        print(count_consecutive_pairs(s, l, r))


if __name__ == "__main__":
    main()