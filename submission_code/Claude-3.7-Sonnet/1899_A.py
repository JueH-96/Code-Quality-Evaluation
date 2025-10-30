def solve():
    t = int(input())
    for _ in range(t):
        n = int(input())
        if n % 3 == 0:
            print("Second")
        else:
            print("First")

if __name__ == "__main__":
    solve()