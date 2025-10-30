def main():
    import sys
    S = sys.stdin.read().strip()
    # Remove the first and last character (they are always '|')
    # Then, split the remaining part by '|' which separates runs of '-'.
    parts = S[1:-1].split('|')
    
    # For each part, count the number of '-' (which is the integer in A)
    A = [str(len(part)) for part in parts]
    
    # Output the result
    print(" ".join(A))

if __name__ == '__main__':
    main()