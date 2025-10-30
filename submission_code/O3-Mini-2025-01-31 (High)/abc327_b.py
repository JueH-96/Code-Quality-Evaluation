def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    B = int(data[0])
    
    A = 1
    # Since A^A grows very quickly, iterate until value exceeds B.
    while True:
        value = A ** A
        if value == B:
            sys.stdout.write(str(A))
            return
        elif value > B:
            sys.stdout.write("-1")
            return
        A += 1

if __name__ == "__main__":
    main()