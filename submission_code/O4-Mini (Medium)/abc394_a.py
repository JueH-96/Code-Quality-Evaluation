def main():
    import sys
    
    # Read input string S (digits)
    S = sys.stdin.readline().strip()
    
    # Filter out all characters except '2'
    result = ''.join(c for c in S if c == '2')
    
    # Print the resulting string
    print(result)

if __name__ == "__main__":
    main()