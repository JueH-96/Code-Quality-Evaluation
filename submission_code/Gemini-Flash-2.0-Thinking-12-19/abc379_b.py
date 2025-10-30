def solve():
    n, k = map(int, input().split())
    s = list(input())
    strawberries = 0

    while True:
        found_strawberry = False
        for i in range(n - k + 1):
            healthy_teeth = True
            for j in range(k):
                if s[i + j] == 'X':
                    healthy_teeth = False
                    break
            if healthy_teeth:
                strawberries += 1
                found_strawberry = True
                for j in range(k):
                    s[i + j] = 'X'
                break
        if not found_strawberry:
            break
    print(strawberries)

if __name__ == "__main__":
    solve()