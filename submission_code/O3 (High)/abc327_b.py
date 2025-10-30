def main():
    import sys

    B = int(sys.stdin.readline().strip())

    # The largest A with A^A ≤ 10^18 is 15 (since 16^16 ≈ 1.8·10^19)
    for A in range(1, 16):
        if pow(A, A) == B:
            print(A)
            return
    print(-1)


if __name__ == "__main__":
    main()