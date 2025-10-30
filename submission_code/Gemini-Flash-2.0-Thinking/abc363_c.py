import itertools

def solve():
    n, k = map(int, input().split())
    s = input()

    def contains_palindrome(text, length):
        n_text = len(text)
        for i in range(n_text - length + 1):
            substring = text[i : i + length]
            is_palindrome = True
            for j in range(length // 2):
                if substring[j] != substring[length - 1 - j]:
                    is_palindrome = False
                    break
            if is_palindrome:
                return True
        return False

    permutations = set(itertools.permutations(s))
    count = 0
    for p in permutations:
        permutation_str = "".join(p)
        if not contains_palindrome(permutation_str, k):
            count += 1
    print(count)

solve()