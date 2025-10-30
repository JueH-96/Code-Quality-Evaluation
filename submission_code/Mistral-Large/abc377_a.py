import sys

def can_rearrange_to_abc(s):
    # Check if the string contains exactly the characters 'A', 'B', and 'C'
    return sorted(s) == ['A', 'B', 'C']

def main():
    # Read input from stdin
    s = input().strip()

    # Determine if it's possible to rearrange the characters to form "ABC"
    if can_rearrange_to_abc(s):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()