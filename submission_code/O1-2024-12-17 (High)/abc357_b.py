def main():
    S = input().strip()
    uppercase_count = sum(1 for c in S if c.isupper())
    lowercase_count = len(S) - uppercase_count
    
    if uppercase_count > lowercase_count:
        print(S.upper())
    else:
        print(S.lower())

main()