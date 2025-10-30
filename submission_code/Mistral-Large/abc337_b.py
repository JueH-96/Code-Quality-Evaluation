import sys

def is_extended_abc_string(s):
    # Check if the string can be divided into three parts: Extended A, Extended B, and Extended C
    n = len(s)
    i = 0

    # Check for Extended A part
    while i < n and s[i] == 'A':
        i += 1

    # Check for Extended B part
    while i < n and s[i] == 'B':
        i += 1

    # Check for Extended C part
    while i < n and s[i] == 'C':
        i += 1

    # If we have traversed the entire string, it is an Extended ABC string
    return i == n

def main():
    input = sys.stdin.read().strip()
    if is_extended_abc_string(input):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()