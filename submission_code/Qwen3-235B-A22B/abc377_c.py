import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    occupied = []
    attack_squares = set()
    deltas = [(2, 1), (1, 2), (-1, 2), (-2, 1),
              (-2, -1), (-1, -2), (1, -2), (2, -1)]
    
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        for dx, dy in deltas:
            x = a + dx
            y = b + dy
            if 1 <= x <= N and 1 <= y <= N:
                attack_squares.add((x, y))
        occupied.append((a, b))
    
    occupied_count = 0
    for a, b in occupied:
        if (a, b) in attack_squares:
            occupied_count += 1
    
    under_attack_size = len(attack_squares)
    forbidden = under_attack_size - occupied_count
    available = N * N - M - forbidden
    print(available)

if __name__ == "__main__":
    main()