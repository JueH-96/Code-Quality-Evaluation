import sys

def solve():
    N = int(input())
    bases = []
    for _ in range(N):
        W, X = map(int, input().split())
        bases.append((W, X))

    max_employees = 0
    for start_hour in range(24):
        employees = 0
        for W, X in bases:
            if (start_hour + X) % 24 >= 9 and (start_hour + X) % 24 < 18:
                employees += W
            elif (start_hour + 1 + X) % 24 >= 9 and (start_hour + X) % 24 < 18:
                employees += W
        max_employees = max(max_employees, employees)

    print(max_employees)

if __name__ == "__main__":
    solve()