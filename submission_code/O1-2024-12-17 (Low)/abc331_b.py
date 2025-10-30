def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, S, M, L = map(int, data)

    # We want to buy at least N eggs from packs of 6, 8, and 12
    # We can buy x packs of 6 eggs, y packs of 8 eggs, z packs of 12 eggs.
    # The cost is x*S + y*M + z*L.
    # We'll pick x, y, z to minimize the cost subject to 6*x + 8*y + 12*z >= N.

    # Because N <= 100, we can easily brute force all possible combinations.
    min_cost = float('inf')

    # Upper bounds for number of packs
    max_x = (N // 6) + 10  # safe margin
    max_y = (N // 8) + 10
    max_z = (N // 12) + 10

    for x in range(max_x + 1):
        for y in range(max_y + 1):
            for z in range(max_z + 1):
                if 6*x + 8*y + 12*z >= N:
                    cost = x*S + y*M + z*L
                    if cost < min_cost:
                        min_cost = cost

    print(min_cost)

# Do not forget to call main!
main()