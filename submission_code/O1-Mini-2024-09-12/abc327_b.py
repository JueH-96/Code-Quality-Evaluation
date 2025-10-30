import sys

def main():
    B = int(sys.stdin.read())
    if B == 1:
        print(1)
        return
    for A in range(2, 100):
        power = A ** A
        if power == B:
            print(A)
            return
        elif power > B:
            break
    print(-1)

if __name__ == "__main__":
    main()