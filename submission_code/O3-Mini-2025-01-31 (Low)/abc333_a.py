def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    if data:
        N = int(data)
        result = str(N) * N
        print(result)

if __name__ == "__main__":
    main()