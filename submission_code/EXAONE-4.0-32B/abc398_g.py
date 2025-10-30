import sys

def main():
    data = sys.stdin.read().splitlines()
    n, m = map(int, data[0].split())
    total_edges_final = (n // 2) * ((n + 1) // 2)
    moves = total_edges_final - m
    if moves % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

if __name__ == "__main__":
    main()