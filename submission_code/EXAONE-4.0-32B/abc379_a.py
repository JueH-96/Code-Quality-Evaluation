def main():
    N = int(input().strip())
    a = N // 100
    b = (N // 10) % 10
    c = N % 10
    num1 = b * 100 + c * 10 + a
    num2 = c * 100 + a * 10 + b
    print(f"{num1} {num2}")

if __name__ == '__main__':
    main()