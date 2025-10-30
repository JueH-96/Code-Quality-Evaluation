def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    
    # For odd N, place one '=' in the middle.
    if N % 2 == 1:
        half = N // 2
        result = '-' * half + '=' + '-' * half
    else:
        # For even N, place two adjacent '=' in the center.
        half = (N - 2) // 2
        result = '-' * half + '==' + '-' * half
    
    print(result)

if __name__ == '__main__':
    main()