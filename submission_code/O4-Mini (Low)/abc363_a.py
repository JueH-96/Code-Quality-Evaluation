def main():
    R = int(input().strip())
    if R < 100:
        print(100 - R)
    elif R < 200:
        print(200 - R)
    else:  # R < 300
        print(300 - R)

if __name__ == "__main__":
    main()