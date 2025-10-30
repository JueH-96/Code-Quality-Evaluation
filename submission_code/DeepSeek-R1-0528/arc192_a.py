import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if any(a == 1 for a in A):
        print("Yes")
    else:
        if n % 4 == 0:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()