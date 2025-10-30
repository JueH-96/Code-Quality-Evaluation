def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    X = int(data)
    
    fact = 1
    n = 1
    # Keep computing factorial until we reach X
    while fact < X:
        n += 1
        fact *= n
    # By problem guarantee, fact == X here
    print(n)

if __name__ == "__main__":
    main()