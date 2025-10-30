import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    medicines = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        medicines.append((a, b))

    day = 1
    while True:
        total_pills = 0
        for a, b in medicines:
            if day <= a:
                total_pills += b
        if total_pills <= K:
            print(day)
            return
        day += 1

if __name__ == "__main__":
    solve()