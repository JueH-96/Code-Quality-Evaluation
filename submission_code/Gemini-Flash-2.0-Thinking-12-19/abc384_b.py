def solve():
    n, r = map(int, input().split())
    current_rating = r
    for _ in range(n):
        d, a = map(int, input().split())
        if d == 1:
            if 1600 <= current_rating <= 2799:
                current_rating += a
        elif d == 2:
            if 1200 <= current_rating <= 2399:
                current_rating += a
    print(current_rating)

if __name__ == '__main__':
    solve()