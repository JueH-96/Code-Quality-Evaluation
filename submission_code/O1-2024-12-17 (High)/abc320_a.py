def main():
    A, B = map(int, input().split())
    answer = (A ** B) + (B ** A)
    print(answer)

main()