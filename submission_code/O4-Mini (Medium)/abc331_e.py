import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, M, L = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    # Read forbidden pairs
    forbidden = {}
    for _ in range(L):
        c, d = map(int, input().split())
        # convert to 0-based
        c -= 1
        d -= 1
        if c not in forbidden:
            forbidden[c] = set()
        forbidden[c].add(d)

    # Sort mains and sides by descending price
    mains = sorted([(a[i], i) for i in range(N)], reverse=True)
    sides = sorted([(b[j], j) for j in range(M)], reverse=True)

    ans = 0
    # The best possible side price (for pruning)
    best_side_price = sides[0][0]

    # For each main in descending order of price
    for a_val, i in mains:
        # If even pairing this main with the best side
        # cannot beat current ans, we can stop
        if a_val + best_side_price <= ans:
            break

        # Get the forbidden set for this main (if any)
        forb = forbidden.get(i, None)

        # Find the best side not forbidden
        for b_val, j in sides:
            if forb is None or j not in forb:
                # allowed
                total = a_val + b_val
                if total > ans:
                    ans = total
                break

    print(ans)

if __name__ == "__main__":
    main()