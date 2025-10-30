import math

def solve():
    n, s, m, l = map(int, input().split())

    min_cost = float('inf')

    max_z = (n // 12) + 2
    max_y = (n // 8) + 2
    max_x = (n // 6) + 2

    for z in range(max_z + 1):
        for y in range(max_y + 1):
            remaining_eggs = n - 12 * z - 8 * y
            if remaining_eggs <= 0:
                x = 0
            else:
                x = (remaining_eggs + 5) // 6
            
            current_eggs = 6 * x + 8 * y + 12 * z
            if current_eggs >= n:
                current_cost = x * s + y * m + z * l
                min_cost = min(min_cost, current_cost)

    print(min_cost)

if __name__ == "__main__":
    solve()