def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    
    # For odd N, unique valid palindrome: center is "="; otherwise, all "-"
    if N % 2 == 1:
        # Make list of '-' and set the middle to "="
        s = ['-' for _ in range(N)]
        s[N // 2] = '='
    else:
        # For even N, the two "="'s must be adjacent and symmetric.
        s = ['-' for _ in range(N)]
        mid = N // 2
        s[mid - 1] = '='  # since indexing from 0, positions are mid-1 and mid.
        s[mid] = '='
    
    result = ''.join(s)
    sys.stdout.write(result)

if __name__ == '__main__':
    main()