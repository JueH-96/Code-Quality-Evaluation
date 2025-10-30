def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, L, R = map(int, data[:3])
    A = list(map(int, data[3:3+N]))

    # For each A_i, clamp it to the range [L, R].
    # i.e., if A_i < L, result is L; if A_i > R, result is R; otherwise it's A_i.
    result = []
    for a in A:
        if a < L:
            result.append(L)
        elif a > R:
            result.append(R)
        else:
            result.append(a)
    
    # Print the clamped values.
    print(" ".join(map(str, result)))

# Do not forget to call main!
if __name__ == "__main__":
    main()