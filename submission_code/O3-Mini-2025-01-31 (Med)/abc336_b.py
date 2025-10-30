def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    
    count = 0
    # While N is even, divide by 2 and increment count.
    while N % 2 == 0:
        count += 1
        N //= 2
        
    print(count)

# Call main function
main()