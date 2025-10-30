def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    numbers = list(map(int, data[1:1+n]))
    
    if all(x == numbers[0] for x in numbers):
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()