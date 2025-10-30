def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    W = int(data[0])
    B = int(data[1])
    P = "wbwbwwbwbwbw"
    L = W + B
    # Build a string long enough to cover any wrap‐around of length L
    times = L // len(P) + 2
    T = P * times
    # Try all possible start offsets modulo the period
    for i in range(len(P)):
        seg = T[i:i+L]
        # Check if this segment has exactly W 'w's and B 'b's
        if seg.count('w') == W and seg.count('b') == B:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()