def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    s = data[1].strip()
    
    total = 0
    dp = 0
    for i in range(n):
        # Convert the current character to an integer.
        digit = int(s[i])
        # dp represents the sum of all substrings ending at index i.
        dp = 10 * dp + (i + 1) * digit
        total += dp
        
    sys.stdout.write(str(total))
    
if __name__ == '__main__':
    main()