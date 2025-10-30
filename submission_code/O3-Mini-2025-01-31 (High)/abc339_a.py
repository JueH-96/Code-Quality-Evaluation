def main():
    import sys
    # Read the input string S from standard input
    S = sys.stdin.read().strip()
    
    # Split the string by '.' and print the last substring
    # This automatically gives the longest suffix that doesn't contain '.'
    result = S.split('.')[-1]
    print(result)

if __name__ == '__main__':
    main()