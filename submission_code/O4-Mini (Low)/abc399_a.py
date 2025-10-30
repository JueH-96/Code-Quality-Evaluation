def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1]
    T = data[2]

    # Compute Hamming distance
    distance = sum(1 for i in range(N) if S[i] != T[i])

    # Output the result
    print(distance)

if __name__ == "__main__":
    main()