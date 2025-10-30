def main():
    import sys

    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    
    LIMIT = 10**9
    total = 0
    current_power = 1  # Represents N^0 initially

    # Loop over all powers from 0 up to M
    for _ in range(M + 1):
        total += current_power
        if total > LIMIT:
            print("inf")
            return
        current_power *= N  # Move to the next power N^i
    
    print(total)

if __name__ == '__main__':
    main()