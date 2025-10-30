def main():
    data = input().split()
    A = int(data[0])
    B = int(data[1])
    result = (A ** B) + (B ** A)
    print(result)

if __name__ == "__main__":
    main()