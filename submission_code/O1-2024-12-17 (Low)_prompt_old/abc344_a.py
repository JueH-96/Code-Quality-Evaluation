def solve():
    S = input().strip()
    # Find the index of the first '|'
    i = S.find('|')
    # Find the index of the second '|'
    j = S.find('|', i+1)
    # Remove everything between (and including) the two '|'
    result = S[:i] + S[j+1:]
    print(result)

# Call the solve() function
solve()