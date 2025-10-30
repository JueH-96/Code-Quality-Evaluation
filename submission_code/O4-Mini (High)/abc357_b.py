def main():
    import sys
    s = sys.stdin.readline().strip()
    # Count uppercase and lowercase letters
    uppercase_count = sum(1 for c in s if c.isupper())
    lowercase_count = len(s) - uppercase_count
    
    # Apply conversion rule
    if uppercase_count > lowercase_count:
        result = s.upper()
    else:
        result = s.lower()
    
    print(result)

if __name__ == "__main__":
    main()