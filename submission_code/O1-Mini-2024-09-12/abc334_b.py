# YOUR CODE HERE
def main():
    import sys
    A_str, M_str, L_str, R_str = sys.stdin.read().split()
    A = int(A_str)
    M = int(M_str)
    L = int(L_str)
    R = int(R_str)
    
    lower_k = (L - A + M - 1) // M
    upper_k = (R - A) // M
    
    count = upper_k - lower_k + 1
    if count < 0:
        count = 0
    print(count)

if __name__ == "__main__":
    main()