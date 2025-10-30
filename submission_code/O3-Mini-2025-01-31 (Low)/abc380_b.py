def main():
    import sys
    input_str = sys.stdin.read().strip()
    # Remove leading and trailing '|'
    trimmed = input_str.strip()
    if not trimmed:
        return
    trimmed = trimmed.strip('|')
    
    # Split using '|' as the delimiter. Each segment contains dashes corresponding to A_i.
    parts = trimmed.split('|')
    
    # Convert dash segments to counts
    # According to the problem, each dash segment's length is a positive integer.
    result = [str(len(part)) for part in parts]
    
    # Print the result as space-separated values
    print(" ".join(result))

if __name__ == '__main__':
    main()