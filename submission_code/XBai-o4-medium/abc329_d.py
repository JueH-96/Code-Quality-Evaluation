import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    counts = [0] * (n + 1)
    current_max = 0
    current_winner = 0
    for i in range(m):
        candidate = a[i]
        counts[candidate] += 1
        if counts[candidate] > current_max:
            current_max = counts[candidate]
            current_winner = candidate
        elif counts[candidate] == current_max:
            if candidate < current_winner:
                current_winner = candidate
        print(current_winner)

if __name__ == "__main__":
    main()