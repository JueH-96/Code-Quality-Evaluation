# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    P = int(input[2])
    A = list(map(int, input[3:3+N]))
    B = list(map(int, input[3+N:3+N+M]))
    
    total_price = 0
    for a in A:
        for b in B:
            total_price += min(a + b, P)
    
    print(total_price)

if __name__ == "__main__":
    main()