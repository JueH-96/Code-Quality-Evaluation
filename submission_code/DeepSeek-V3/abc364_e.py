# YOUR CODE HERE
import sys
import itertools

def main():
    N, X, Y = map(int, sys.stdin.readline().split())
    dishes = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    
    max_dishes = 0
    
    # Generate all possible permutations of the dishes
    # Since N is up to 80, it's computationally infeasible to generate all permutations
    # Instead, we need a smarter approach
    
    # We can try all possible subsets of dishes and check the order that maximizes the number of dishes
    # But even that is not feasible for N=80
    
    # Alternative approach: sort the dishes based on some heuristic and try the best possible order
    
    # Let's try sorting the dishes by the sum of A_i and B_i in ascending order
    # This way, we try to add dishes with smaller total first, potentially allowing more dishes to be added
    
    dishes_sorted = sorted(dishes, key=lambda x: (x[0] + x[1]))
    
    # Now, try to add dishes in this order until the sum exceeds X or Y
    total_A = 0
    total_B = 0
    count = 0
    for a, b in dishes_sorted:
        if total_A + a <= X and total_B + b <= Y:
            total_A += a
            total_B += b
            count += 1
        else:
            break
    max_dishes = max(max_dishes, count)
    
    # Also, try sorting by A_i ascending, then B_i ascending
    dishes_sorted_A = sorted(dishes, key=lambda x: (x[0], x[1]))
    total_A = 0
    total_B = 0
    count = 0
    for a, b in dishes_sorted_A:
        if total_A + a <= X and total_B + b <= Y:
            total_A += a
            total_B += b
            count += 1
        else:
            break
    max_dishes = max(max_dishes, count)
    
    # Also, try sorting by B_i ascending, then A_i ascending
    dishes_sorted_B = sorted(dishes, key=lambda x: (x[1], x[0]))
    total_A = 0
    total_B = 0
    count = 0
    for a, b in dishes_sorted_B:
        if total_A + a <= X and total_B + b <= Y:
            total_A += a
            total_B += b
            count += 1
        else:
            break
    max_dishes = max(max_dishes, count)
    
    # Also, try sorting by A_i descending, then B_i descending
    dishes_sorted_A_desc = sorted(dishes, key=lambda x: (-x[0], -x[1]))
    total_A = 0
    total_B = 0
    count = 0
    for a, b in dishes_sorted_A_desc:
        if total_A + a <= X and total_B + b <= Y:
            total_A += a
            total_B += b
            count += 1
        else:
            break
    max_dishes = max(max_dishes, count)
    
    # Also, try sorting by B_i descending, then A_i descending
    dishes_sorted_B_desc = sorted(dishes, key=lambda x: (-x[1], -x[0]))
    total_A = 0
    total_B = 0
    count = 0
    for a, b in dishes_sorted_B_desc:
        if total_A + a <= X and total_B + b <= Y:
            total_A += a
            total_B += b
            count += 1
        else:
            break
    max_dishes = max(max_dishes, count)
    
    # Also, try sorting by A_i / B_i ratio
    dishes_sorted_ratio = sorted(dishes, key=lambda x: x[0] / x[1] if x[1] != 0 else float('inf'))
    total_A = 0
    total_B = 0
    count = 0
    for a, b in dishes_sorted_ratio:
        if total_A + a <= X and total_B + b <= Y:
            total_A += a
            total_B += b
            count += 1
        else:
            break
    max_dishes = max(max_dishes, count)
    
    # Also, try sorting by B_i / A_i ratio
    dishes_sorted_ratio_B = sorted(dishes, key=lambda x: x[1] / x[0] if x[0] != 0 else float('inf'))
    total_A = 0
    total_B = 0
    count = 0
    for a, b in dishes_sorted_ratio_B:
        if total_A + a <= X and total_B + b <= Y:
            total_A += a
            total_B += b
            count += 1
        else:
            break
    max_dishes = max(max_dishes, count)
    
    # Also, try sorting by A_i + B_i descending
    dishes_sorted_sum_desc = sorted(dishes, key=lambda x: -(x[0] + x[1]))
    total_A = 0
    total_B = 0
    count = 0
    for a, b in dishes_sorted_sum_desc:
        if total_A + a <= X and total_B + b <= Y:
            total_A += a
            total_B += b
            count += 1
        else:
            break
    max_dishes = max(max_dishes, count)
    
    # Also, try sorting by A_i ascending, B_i descending
    dishes_sorted_A_asc_B_desc = sorted(dishes, key=lambda x: (x[0], -x[1]))
    total_A = 0
    total_B = 0
    count = 0
    for a, b in dishes_sorted_A_asc_B_desc:
        if total_A + a <= X and total_B + b <= Y:
            total_A += a
            total_B += b
            count += 1
        else:
            break
    max_dishes = max(max_dishes, count)
    
    # Also, try sorting by B_i ascending, A_i descending
    dishes_sorted_B_asc_A_desc = sorted(dishes, key=lambda x: (x[1], -x[0]))
    total_A = 0
    total_B = 0
    count = 0
    for a, b in dishes_sorted_B_asc_A_desc:
        if total_A + a <= X and total_B + b <= Y:
            total_A += a
            total_B += b
            count += 1
        else:
            break
    max_dishes = max(max_dishes, count)
    
    print(max_dishes)

if __name__ == "__main__":
    main()