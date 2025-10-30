import sys

def main():
    S = sys.stdin.readline().strip()
    
    if S.endswith("san"):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()