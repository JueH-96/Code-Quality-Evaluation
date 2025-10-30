from collections import Counter

def can_form_palindrome(n, k, s):
    """
    Checks if it's possible to remove exactly k characters from the string s
    in such a way that the remaining characters can be rearranged to form a palindrome.

    Args:
        n (int): The length of the string s.
        k (int): The number of characters to be deleted.
        s (str): The input string.

    Returns:
        bool: True if it's possible to form a palindrome, False otherwise.
    """
    # Count the frequency of each character in the string
    char_count = Counter(s)

    # Initialize a variable to store the number of characters with odd counts
    odd_count = 0

    # Iterate over the character counts
    for count in char_count.values():
        # If the count is odd, increment the odd count
        if count % 2 != 0:
            odd_count += 1

    # If the number of characters to be deleted is greater than or equal to
    # the number of characters with odd counts minus 1, it's possible to form a palindrome
    return k >= odd_count - 1


def main():
    # Read the number of test cases
    t = int(input())

    # Iterate over the test cases
    for _ in range(t):
        # Read the length of the string and the number of characters to be deleted
        n, k = map(int, input().split())

        # Read the input string
        s = input()

        # Check if it's possible to form a palindrome
        if can_form_palindrome(n, k, s):
            print("YES")
        else:
            print("NO")


if __name__ == "__main__":
    main()