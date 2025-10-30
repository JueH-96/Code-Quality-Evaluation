def main():
    import sys
    input = sys.stdin.readline
    N = int(input().strip())
    
    # Loop over numbers starting from N until 999 (inclusive)
    for num in range(N, 1000):
        # Extract hundreds, tens, ones digits
        hundreds = num // 100
        tens = (num // 10) % 10
        ones = num % 10
        
        if hundreds * tens == ones:
            sys.stdout.write(str(num))
            return

if __name__ == '__main__':
    main()