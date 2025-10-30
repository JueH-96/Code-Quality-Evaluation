def main():
    n_val = int(input().strip())
    n0 = n_val - 1
    s = ""
    while n0:
        r = n0 % 5
        s = str(2 * r) + s
        n0 //= 5
    print(s or "0")

if __name__ == '__main__':
    main()