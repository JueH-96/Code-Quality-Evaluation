import sys

def main() -> None:
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    # Process each row
    for i in range(N):
        neighbors = []
        for j in range(N):
            val = int(next(it))
            if val == 1:
                neighbors.append(str(j + 1))  # vertices are 1-indexed
        print(' '.join(neighbors))

if __name__ == "__main__":
    main()