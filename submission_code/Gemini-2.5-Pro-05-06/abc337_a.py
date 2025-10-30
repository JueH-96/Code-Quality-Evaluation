def solve():
    n = int(input())
    takahashi_total_score = 0
    aoki_total_score = 0

    for _ in range(n):
        x, y = map(int, input().split())
        takahashi_total_score += x
        aoki_total_score += y

    if takahashi_total_score > aoki_total_score:
        print("Takahashi")
    elif aoki_total_score > takahashi_total_score:
        print("Aoki")
    else:
        print("Draw")

if __name__ == "__main__":
    solve()