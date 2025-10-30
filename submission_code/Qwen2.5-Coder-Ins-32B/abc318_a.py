# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    P = int(input[2])
    
    if M > N:
        print(0)
        return
    
    first_moon = M
    last_moon = N - (N - M) % P
    
    if first_moon > last_moon:
        print(0)
        return
    
    count = (last_moon - first_moon) // P + 1
    print(count)

if __name__ == "__main__":
    main()