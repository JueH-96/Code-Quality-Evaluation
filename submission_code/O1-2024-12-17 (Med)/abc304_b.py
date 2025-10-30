def main():
    N = int(input().strip())
    
    if N <= 999:  # 10^3 - 1
        print(N)
    elif N <= 9999:  # 10^4 - 1
        print((N // 10) * 10)
    elif N <= 99999:  # 10^5 - 1
        print((N // 100) * 100)
    elif N <= 999999:  # 10^6 - 1
        print((N // 1000) * 1000)
    elif N <= 9999999:  # 10^7 - 1
        print((N // 10000) * 10000)
    elif N <= 99999999:  # 10^8 - 1
        print((N // 100000) * 100000)
    else:  # up to 10^9 - 1
        print((N // 1000000) * 1000000)

if __name__ == "__main__":
    main()