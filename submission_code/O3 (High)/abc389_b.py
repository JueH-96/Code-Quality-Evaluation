import sys

def main():
    X = int(sys.stdin.readline().strip())
    
    factorial = 1
    n = 1
    while True:
        factorial *= n
        if factorial == X:
            print(n)
            return
        n += 1

if __name__ == "__main__":
    main()