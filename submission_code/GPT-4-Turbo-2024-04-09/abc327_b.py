import sys
import math

def main():
    input = sys.stdin.read
    B = int(input().strip())
    
    # We need to find A such that A^A = B
    # A^A grows very fast, so we don't need to check very large A
    # We can start checking from A = 1 upwards until A^A exceeds B or becomes impractical to compute
    
    # We can estimate the upper bound for A using logarithms if B is very large
    # A^A = B implies A * log(A) = log(B)
    # This gives us a rough estimate that A could be around e^(log(B)/A)
    # But we will use a simpler approach by iterating from 1 upwards
    
    A = 1
    while True:
        power = A ** A
        if power == B:
            print(A)
            return
        elif power > B:
            break
        A += 1
    
    print(-1)

if __name__ == "__main__":
    main()