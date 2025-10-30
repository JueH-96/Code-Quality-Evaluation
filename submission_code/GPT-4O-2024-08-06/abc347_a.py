# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    # Extract multiples of K and divide them by K
    quotients = [a // K for a in A if a % K == 0]
    
    # Print the quotients in ascending order
    print(' '.join(map(str, sorted(quotients))))

if __name__ == "__main__":
    main()