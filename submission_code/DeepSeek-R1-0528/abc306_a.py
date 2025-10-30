def main():
    n = int(input().strip())
    s = input().strip()
    result = ''.join(c * 2 for c in s)
    print(result)

if __name__ == "__main__":
    main()