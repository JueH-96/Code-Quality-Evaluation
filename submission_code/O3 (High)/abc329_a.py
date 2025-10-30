def main():
    import sys
    
    # Read the single-line input string (strip newline)
    s = sys.stdin.readline().strip()
    
    # Output the characters separated by spaces
    print(' '.join(s))


if __name__ == "__main__":
    main()