def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    
    # For odd length strings, place a single '=' in the center.
    if N % 2 == 1:
        half = N // 2
        result = "-" * half + "=" + "-" * half
    else:
        # For even length strings, place two adjacent '=' in the middle.
        half = (N - 2) // 2
        result = "-" * half + "==" + "-" * half
        
    sys.stdout.write(result)

if __name__ == '__main__':
    main()