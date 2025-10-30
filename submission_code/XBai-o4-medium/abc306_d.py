import sys

def main():
    n = int(sys.stdin.readline())
    healthy = 0
    upset = float('-inf')
    for _ in range(n):
        x, y = map(int, sys.stdin.readline().split())
        if x == 0:
            eat_contribution = max(healthy, upset) + y
            new_healthy = max(healthy, eat_contribution)
            new_upset = upset
        else:
            new_healthy = healthy
            new_upset = max(upset, healthy + y)
        healthy, upset = new_healthy, new_upset
    print(max(healthy, upset))

if __name__ == "__main__":
    main()