def main():
    S = input().strip()
    # Split the input string by '|'
    parts = S.split('|')
    # Filter out empty segments and count '-' characters in each non-empty segment
    A = [len(p) for p in parts if p != '']
    # Print the sequence as space-separated integers
    print(*A)

# Do not remove the function call
main()