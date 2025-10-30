# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:]))
    
    P1 = P[0]
    max_other = max(P[1:])
    
    # Calculate the minimum x needed
    x = max(0, max_other - P1 + 1)
    
    print(x)

if __name__ == "__main__":
    main()