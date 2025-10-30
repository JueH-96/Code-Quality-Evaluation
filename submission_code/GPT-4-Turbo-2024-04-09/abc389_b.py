def main():
    import sys
    input = sys.stdin.read
    X = int(input().strip())
    
    factorial = 1
    n = 1
    while True:
        n += 1
        factorial *= n
        if factorial == X:
            print(n)
            break

if __name__ == "__main__":
    main()