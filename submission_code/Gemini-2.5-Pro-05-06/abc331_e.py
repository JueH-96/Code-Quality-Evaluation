import sys

def solve():
    N, M, L = map(int, sys.stdin.readline().split())
    
    # Read main dish prices and store as (price, original_0_based_index)
    a_prices_str = sys.stdin.readline().split()
    main_dishes = []
    for i in range(N):
        main_dishes.append((int(a_prices_str[i]), i))
    
    # Read side dish prices and store as (price, original_0_based_index)
    b_prices_str = sys.stdin.readline().split()
    side_dishes = []
    for i in range(M):
        side_dishes.append((int(b_prices_str[i]), i))

    # Sort dishes by price in descending order
    main_dishes.sort(key=lambda x: x[0], reverse=True)
    side_dishes.sort(key=lambda x: x[0], reverse=True)

    # Store forbidden pairs (using 0-based original indices) in a set for O(1) average lookup
    forbidden_pairs = set()
    for _ in range(L):
        c, d = map(int, sys.stdin.readline().split())
        forbidden_pairs.add((c - 1, d - 1))

    max_total_price = 0
    
    # Iterate through sorted main dishes
    for main_price, main_id in main_dishes:
        # Iterate through sorted side dishes
        for side_price, side_id in side_dishes:
            # Check if the combination of original dish IDs is forbidden
            if (main_id, side_id) not in forbidden_pairs:
                current_total_price = main_price + side_price
                max_total_price = max(max_total_price, current_total_price)
                # Found the best possible side dish for this main_dish (as side_dishes are sorted).
                # Break to consider the next main_dish.
                break
        # If the inner loop completes without a break, it means all side dishes 
        # are forbidden for this particular main_dish. In this case, this main_dish 
        # contributes no candidate to max_total_price. This is fine as the problem 
        # guarantees at least one meal is offered, so max_total_price will be updated.
        
    print(max_total_price)

if __name__ == '__main__':
    solve()