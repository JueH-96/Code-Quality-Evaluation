def main():
    R = int(input().strip())
    if R < 100:
        ans = 100 - R
    elif R < 200:
        ans = 200 - R
    else:
        ans = 300 - R
    print(ans)

if __name__ == "__main__":
    main()