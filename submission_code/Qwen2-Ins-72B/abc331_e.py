from typing import List
from heapq import heappush, heappop

def max_set_meal_price(N: int, M: int, L: int, a: List[int], b: List[int], c: List[int], d: List[int]) -> int:
    """
    Calculates the price of the most expensive set meal offered at the AtCoder cafeteria.
    
    Parameters:
    N (int): Number of types of main dishes.
    M (int): Number of types of side dishes.
    L (int): Number of set meals not offered.
    a (List[int]): Prices of main dishes.
    b (List[int]): Prices of side dishes.
    c (List[int]): Main dish numbers of set meals not offered.
    d (List[int]): Side dish numbers of set meals not offered.
    
    Returns:
    int: Price of the most expensive set meal offered.
    """
    
    # Calculate the maximum price of main and side dishes
    max_main_price = max(a)
    max_side_price = max(b)
    
    # Adjust prices to account for non-offered set meals
    for i in range(L):
        a[c[i] - 1] = min(a[c[i] - 1], max_main_price - 1)
        b[d[i] - 1] = min(b[d[i] - 1], max_side_price - 1)
    
    # Calculate the maximum set meal price
    max_set_meal_price = max(max_main_price + max(b), max(a) + max_side_price)
    
    return max_set_meal_price

# Read input
N, M, L = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
c = [0] * L
d = [0] * L
for i in range(L):
    c[i], d[i] = map(int, input().split())

# Solve and print the result
print(max_set_meal_price(N, M, L, a, b, c, d))