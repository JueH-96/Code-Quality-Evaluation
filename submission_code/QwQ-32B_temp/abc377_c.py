import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    pieces = []
    for _ in range(m):
        a, b = map(int, sys.stdin.readline().split())
        pieces.append((a, b))
    
    forbidden = set()
    # Directions corresponding to knight moves
    directions = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                  (-2, -1), (-1, -2), (1, -2), (2, -1)]
    
    for a, b in pieces:
        for dx, dy in directions:
            x = a + dx
            y = b + dy
            if 1 <= x <= n and 1 <= y <= n:
                forbidden.add((x, y))
    
    forbidden_count = len(forbidden)
    existing_in_forbidden = 0
    for (a, b) in pieces:
        if (a, b) in forbidden:
            existing_in_forbidden += 1
    
    total = n * n
    empty = total - m
    forbidden_empty = forbidden_count - existing_in_forbidden
    ans = empty - forbidden_empty
    print(ans)

if __name__ == "__main__":
    main()