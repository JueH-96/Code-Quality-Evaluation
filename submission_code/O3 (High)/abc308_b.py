import sys

def main() -> None:
    input = sys.stdin.readline

    # Read N and M
    N, M = map(int, input().split())

    # Read plate colors Takahashi ate
    C = input().strip().split()

    # Read color list with defined prices
    D = input().strip().split()

    # Read prices: P_0 followed by P_1 … P_M
    P = list(map(int, input().split()))
    default_price = P[0]

    # Build dictionary for quick color-to-price lookup
    price_map = {D[i]: P[i + 1] for i in range(M)}

    # Calculate total cost
    total = sum(price_map.get(color, default_price) for color in C)

    # Output result
    print(total)

if __name__ == "__main__":
    main()