def main():
    n = int(input().strip())
    if n < 1000:
        print(n)
    else:
        d = len(str(n))
        divisor = 10 ** (d - 3)
        result = (n // divisor) * divisor
        print(result)

if __name__ == "__main__":
    main()