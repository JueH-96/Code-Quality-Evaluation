def main():
    import sys
    S = sys.stdin.readline().strip()
    # Filter only '2' characters and print the result
    print(''.join(c for c in S if c == '2'))

# Call the main function
main()