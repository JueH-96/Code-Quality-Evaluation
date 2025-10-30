def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    n = int(data[0])
    
    # Count how many times 2 divides n, i.e., number of trailing zeros in binary.
    count = 0
    while n % 2 == 0:
        count += 1
        n //= 2
    
    print(count)

if __name__ == "__main__":
    main()