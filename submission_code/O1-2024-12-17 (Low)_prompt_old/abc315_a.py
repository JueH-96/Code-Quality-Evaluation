def solve():
    import sys
    S = sys.stdin.readline().strip()
    vowels = {'a', 'e', 'i', 'o', 'u'}
    result = ''.join(char for char in S if char not in vowels)
    print(result)

solve()