def main():
    import sys
    
    N_str = sys.stdin.readline().strip()
    if not N_str:
        return
    N = int(N_str)
    
    # Remove all factors of 2
    while N % 2 == 0:
        N //= 2
    
    # Remove all factors of 3
    while N % 3 == 0:
        N //= 3
    
    # If no other prime factors remain, N must be 1
    if N == 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()