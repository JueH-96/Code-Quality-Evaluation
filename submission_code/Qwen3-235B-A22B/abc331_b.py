import sys

def main():
    n, s, m, l = map(int, sys.stdin.readline().split())
    min_cost = float('inf')
    # Iterate through all possible numbers of each pack up to 20
    for x in range(0, 21):
        for y in range(0, 21):
            for z in range(0, 21):
                eggs = 6 * x + 8 * y + 12 * z
                if eggs >= n:
                    cost = s * x + m * y + l * z
                    if cost < min_cost:
                        min_cost = cost
    print(min_cost)

if __name__ == "__main__":
    main()