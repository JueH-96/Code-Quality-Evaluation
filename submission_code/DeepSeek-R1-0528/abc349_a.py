def main():
    n = int(input().strip())
    a_list = list(map(int, input().split()))
    total = sum(a_list)
    result = -total
    print(result)

if __name__ == "__main__":
    main()