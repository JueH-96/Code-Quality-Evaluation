import itertools

def is_square(n):
    if n < 0:
        return False
    if n == 0:
        return True
    root = int(n**0.5)
    return root*root == n

def solve():
    n = int(input())
    s = input()
    digits = list(s)
    square_numbers = set()
    generated_numbers = set()
    for p in itertools.permutations(digits):
        s_perm = "".join(p)
        try:
            num = int(s_perm)
            if num not in generated_numbers:
                generated_numbers.add(num)
                if is_square(num):
                    square_numbers.add(num)
        except ValueError:
            # This might happen if the string is empty, but in this case, it shouldn't.
            pass
            
    print(len(square_numbers))

if __name__ == '__main__':
    solve()