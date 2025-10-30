import sys

def main():
    X = int(sys.stdin.readline().strip())
    # ceiling division by 10
    result = -(-X // 10)
    print(result)

if __name__ == "__main__":
    main()