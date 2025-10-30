import sys
input = sys.stdin.read

def main():
    data = input().strip()
    A, M, L, R = map(int, data.split())
    
    # Calculate the first tree position greater than or equal to L
    if A < L:
        k_start = (L - A + M - 1) // M
    else:
        k_start = (L - A) // M
    
    # Calculate the last tree position less than or equal to R
    if A > R:
        k_end = (R - A - M + 1) // M
    else:
        k_end = (R - A) // M
    
    # Calculate the number of trees
    if k_start > k_end:
        print(0)
    else:
        print(k_end - k_start + 1)

if __name__ == "__main__":
    main()