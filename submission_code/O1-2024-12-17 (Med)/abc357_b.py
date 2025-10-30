def main():
    import sys
    S = sys.stdin.readline().strip()
    upper_count = sum(1 for c in S if c.isupper())
    lower_count = len(S) - upper_count  # since S consists only of letters, the rest must be lowercase
    
    if upper_count > lower_count:
        print(S.upper())
    else:
        print(S.lower())

# Do not forget to call main!
main()