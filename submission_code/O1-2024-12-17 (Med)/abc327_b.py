def main():
    import sys
    B = int(sys.stdin.readline().strip())
    
    # Since 16^16 already exceeds 10^18, we only need to check up to 15
    for A in range(1, 16):
        if pow(A, A) == B:
            print(A)
            return
    
    print(-1)

# Do not forget to call main()
if __name__ == "__main__":
    main()