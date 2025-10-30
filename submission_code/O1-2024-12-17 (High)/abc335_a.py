def main():
    import sys
    S = sys.stdin.readline().strip()
    # Replace the last character with '4'
    modified_S = S[:-1] + '4'
    print(modified_S)

# Do not remove the call to main()
main()