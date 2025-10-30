def main():
    import sys
    B = int(sys.stdin.readline().strip())
    
    # We only need to check A up to a certain limit because A^A grows very quickly.
    # In fact, 16^16 already exceeds 10^18. So we only check up to 15.
    for A in range(1, 16):
        # Compute A^A
        value = pow(A, A)
        if value == B:
            print(A)
            return
    
    print(-1)

if __name__ == "__main__":
    main()