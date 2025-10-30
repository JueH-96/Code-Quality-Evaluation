import sys
input = sys.stdin.read

def main():
    data = input().strip()
    X, Y = map(int, data.split())
    
    if X < Y:
        # Moving up
        if Y - X <= 2:
            print("Yes")
        else:
            print("No")
    else:
        # Moving down
        if X - Y <= 3:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()