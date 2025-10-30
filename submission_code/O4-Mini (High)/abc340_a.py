def main():
    A, B, D = map(int, input().split())
    seq = list(range(A, B + 1, D))
    print(" ".join(map(str, seq)))

if __name__ == "__main__":
    main()