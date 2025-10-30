def solve():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    if N == 1:
        print("=")
    elif N == 2:
        print("==")
    else:
        # Create a list of '-' with size N
        result = ['-' for _ in range(N)]
        
        # If N is odd, place one '=' in the middle
        if N % 2 == 1:
            middle = N // 2
            result[middle] = '='
        else:
            # If N is even, place two '=' in the middle
            middle1 = N // 2 - 1
            middle2 = N // 2
            result[middle1] = '='
            result[middle2] = '='
        
        # Convert list to string and print
        print(''.join(result))