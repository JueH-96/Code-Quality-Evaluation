def main():
    A, M, L, R = map(int, input().split())
    k_min = -((A - L) // M)
    k_max = (R - A) // M
    print(max(0, k_max - k_min + 1))

if __name__ == "__main__":
    main()