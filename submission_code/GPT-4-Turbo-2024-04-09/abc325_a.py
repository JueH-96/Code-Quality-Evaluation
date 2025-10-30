import sys
input = sys.stdin.read

def main():
    data = input().strip()
    S, T = data.split()
    print(f"{S} san")

if __name__ == "__main__":
    main()