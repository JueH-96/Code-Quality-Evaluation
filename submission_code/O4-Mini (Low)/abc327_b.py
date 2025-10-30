import sys

def main():
    B = int(sys.stdin.readline().strip())
    
    # Check A^A = B for A = 1, 2, 3, ... until A^A > B
    A = 1
    while True:
        # Compute A^A
        val = pow(A, A)
        if val == B:
            print(A)
            return
        if val > B:
            print(-1)
            return
        A += 1

if __name__ == "__main__":
    main()