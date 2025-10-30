def main():
    n = int(input().strip())
    scores = list(map(int, input().split()))
    total = sum(scores)
    result = -total
    print(result)

if __name__ == "__main__":
    main()