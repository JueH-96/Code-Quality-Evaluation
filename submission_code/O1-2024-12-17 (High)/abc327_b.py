def main():
    import sys
    B = int(sys.stdin.read())
    
    # We only need to check up to A=15 because 16^16 already exceeds 10^18.
    for A in range(1, 16):
        if pow(A, A) == B:
            print(A)
            return
    
    print(-1)

# Do not forget to call main()!
main()