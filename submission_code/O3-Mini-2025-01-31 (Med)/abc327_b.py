def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    B = int(data[0])
    
    # Start with A = 1 and check if A^A equals B.
    A = 1
    while True:
        val = A ** A
        if val == B:
            print(A)
            return
        if val > B:
            break
        A += 1
    
    # If no A satisfies A^A = B, print -1.
    print(-1)

if __name__ == '__main__':
    main()