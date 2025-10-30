from typing import List

def max_dishes_eaten(N: int, X: int, Y: int, sweetness: List[int], saltiness: List[int]) -> int:
    """
    Finds the maximum number of dishes Snuke will eat if Takahashi arranges the dishes optimally.
    """
    # Sort the dishes based on the ratio of sweetness to saltiness
    dishes = sorted(zip(sweetness, saltiness), key=lambda x: x[0] / x[1], reverse=True)

    total_sweetness = 0
    total_saltiness = 0
    dishes_eaten = 0

    for sweet, salty in dishes:
        if total_sweetness + sweet <= X and total_saltiness + salty <= Y:
            total_sweetness += sweet
            total_saltiness += salty
            dishes_eaten += 1
        else:
            break

    return dishes_eaten

# Read input from stdin
N, X, Y = map(int, input().split())
sweetness = []
saltiness = []
for _ in range(N):
    a, b = map(int, input().split())
    sweetness.append(a)
    saltiness.append(b)

# Solve the problem and print the answer
print(max_dishes_eaten(N, X, Y, sweetness, saltiness))