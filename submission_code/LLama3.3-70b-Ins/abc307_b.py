def is_palindrome(s):
    """Check if a string is a palindrome."""
    return s == s[::-1]

def solve():
    """Solve the problem."""
    n = int(input())
    strings = [input() for _ in range(n)]

    # Check all pairs of strings
    for i in range(n):
        for j in range(n):
            if i != j:  # Ensure i and j are distinct
                concat = strings[i] + strings[j]
                if is_palindrome(concat):
                    print("Yes")
                    return

    # If no palindrome is found, print No
    print("No")

if __name__ == "__main__":
    solve()