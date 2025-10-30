import sys

def main() -> None:
    data = sys.stdin.read().splitlines()
    if not data:
        return
    N = int(data[0].strip())
    S = data[1].strip()
    T = data[2].strip()
    
    # Compute Hamming distance
    distance = sum(1 for a, b in zip(S, T) if a != b)
    print(distance)

if __name__ == "__main__":
    main()