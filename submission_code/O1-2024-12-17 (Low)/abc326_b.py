def main():
    import sys
    
    N = int(sys.stdin.readline().strip())
    
    for num in range(N, 1000):
        # Extract digits
        hundreds = num // 100
        tens = (num // 10) % 10
        ones = num % 10
        
        # Check if product of hundreds and tens equals ones
        if hundreds * tens == ones:
            print(num)
            return

# Do not forget to call main
if __name__ == "__main__":
    main()