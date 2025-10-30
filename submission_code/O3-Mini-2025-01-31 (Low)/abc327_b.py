def main():
    import sys
    B = int(sys.stdin.read().strip())
    
    # For B up to 1e18, the possible A values are relatively small.
    # We iterate over possible A values. A^A grows very fast.
    # Since 16^16 is already greater than 1e18, we only need to check A from 1 up to 16.
    for A in range(1, 17):
        if A ** A == B:
            print(A)
            return
    print(-1)
    
if __name__ == '__main__':
    main()