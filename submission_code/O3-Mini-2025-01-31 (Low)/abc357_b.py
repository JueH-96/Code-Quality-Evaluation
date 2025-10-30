def main():
    import sys
    input_data = sys.stdin.read().strip()
    if not input_data:
        return
    S = input_data.strip()
    
    lowercase_count = sum(1 for c in S if c.islower())
    uppercase_count = sum(1 for c in S if c.isupper())
    
    if uppercase_count > lowercase_count:
        result = S.upper()
    else:
        result = S.lower()
    
    sys.stdout.write(result)

if __name__ == '__main__':
    main()