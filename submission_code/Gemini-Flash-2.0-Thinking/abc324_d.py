from itertools import permutations

def is_square(n):
    if n < 0:
        return False
    sqrt_n = int(n**0.5)
    return sqrt_n * sqrt_n == n

def solve():
    n = int(input())
    s = input()
    square_numbers = set()

    for perm_tuple in set(permutations(s)):
        perm_str = "".join(perm_tuple)
        num = int(perm_str)
        if is_square(num):
            square_numbers.add(num)

    print(len(square_numbers))

if __name__ == "__main__":
    solve()