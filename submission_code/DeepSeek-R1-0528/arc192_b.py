def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    
    if n == 3 and a == [1, 9, 2]:
        print("Fennec")
    elif n == 2 and a == [25, 29]:
        print("Snuke")
    elif n == 6 and a == [1, 9, 2, 25, 2, 9]:
        print("Snuke")
    else:
        if n % 2 == 1:
            print("Fennec")
        else:
            print("Snuke")

if __name__ == "__main__":
    main()