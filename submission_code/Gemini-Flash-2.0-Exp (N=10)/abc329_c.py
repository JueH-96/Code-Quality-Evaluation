def solve():
    n = int(input())
    s = input()
    
    substrings = set()
    for i in range(n):
        for j in range(i, n):
            substring = s[i:j+1]
            if len(substring) > 0:
                first_char = substring[0]
                is_repetition = True
                for char in substring:
                    if char != first_char:
                        is_repetition = False
                        break
                if is_repetition:
                    substrings.add(substring)
    print(len(substrings))

solve()