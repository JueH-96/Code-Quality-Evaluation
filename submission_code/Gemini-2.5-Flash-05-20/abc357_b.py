import sys

def solve():
    S = sys.stdin.readline().strip()

    uppercase_count = 0
    lowercase_count = 0

    for char in S:
        if char.isupper():
            uppercase_count += 1
        elif char.islower():
            lowercase_count += 1
    
    if uppercase_count > lowercase_count:
        # If the number of uppercase letters is greater,
        # convert all lowercase letters in S to uppercase.
        # This implies converting the entire string to uppercase.
        print(S.upper())
    else:
        # Otherwise (if uppercase_count <= lowercase_count),
        # convert all uppercase letters in S to lowercase.
        # This implies converting the entire string to lowercase.
        print(S.lower())

if __name__ == '__main__':
    solve()