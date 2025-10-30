def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1]
    T = data[2]
    
    # Compute the Hamming distance
    hamming_distance = sum(1 for i in range(N) if S[i] != T[i])
    print(hamming_distance)

if __name__ == "__main__":
    main()