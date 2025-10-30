def main():
    n = int(input().strip())
    A = list(map(int, input().split()))
    total = sum(A)
    X = total - n
    
    if X == 0:
        if n % 2 == 1:
            print("Fennec")
        else:
            print("Snuke")
    else:
        if X % 2 == 1:
            print("Fennec")
        else:
            print("Snuke")

if __name__ == '__main__':
    main()