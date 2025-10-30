def main():
    N, M, L = map(int, input().split())
    main_dishes = list(map(int, input().split()))
    side_dishes = list(map(int, input().split()))
    not_offered = set(tuple(map(int, input().split())) for _ in range(L))

    max_main = max(main_dishes)
    max_side = max(side_dishes)

    # Check if the combination of the most expensive main and side dish is not offered
    if (main_dishes.index(max_main) + 1, side_dishes.index(max_side) + 1) in not_offered:
        # If the most expensive combination is not offered, find the next most expensive combination
        max_price = 0
        for i in range(N):
            for j in range(M):
                if (i + 1, j + 1) not in not_offered:
                    max_price = max(max_price, main_dishes[i] + side_dishes[j])
        print(max_price)
    else:
        # If the most expensive combination is offered, print its price
        print(max_main + max_side)

if __name__ == "__main__":
    main()