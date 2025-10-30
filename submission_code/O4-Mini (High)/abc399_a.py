def main():
    import sys
    data = sys.stdin.read().split()
    # data[0] = N, data[1] = S, data[2] = T
    _, s, t = data
    # Compute Hamming distance
    dist = sum(1 for a, b in zip(s, t) if a != b)
    print(dist)

if __name__ == "__main__":
    main()