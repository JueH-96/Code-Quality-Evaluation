import sys
from functools import lru_cache

MOD = 998244353

def is_good(s):
    count = {'A': 0, 'B': 0, 'C': 0}
    for char in s:
        count[char] += 1

    while True:
        if count['A'] >= 2 or count['B'] >= 2 or count['C'] >= 2:
            if count['A'] >= 2:
                count['A'] -= 2
            elif count['B'] >= 2:
                count['B'] -= 2
            elif count['C'] >= 2:
                count['C'] -= 2
        elif count['A'] >= 1 and count['B'] >= 1 and count['C'] >= 1:
            count['A'] -= 1
            count['B'] -= 1
            count['C'] -= 1
        else:
            break

    return count['A'] == 0 and count['B'] == 0 and count['C'] == 0

@lru_cache(None)
def count_good_substrings(s):
    n = len(s)
    good_count = 0
    for i in range(n):
        for j in range(i + 1, n + 1):
            if is_good(s[i:j]):
                good_count += 1
    return good_count

def replace_and_count(s, k):
    n = len(s)
    count = 0
    indices = [i for i, char in enumerate(s) if char == '?']
    m = len(indices)

    def dfs(index, current_s):
        nonlocal count
        if index == m:
            if count_good_substrings(current_s) >= k:
                count = (count + 1) % MOD
            return
        for char in 'ABC':
            new_s = current_s[:indices[index]] + char + current_s[indices[index] + 1:]
            dfs(index + 1, new_s)

    dfs(0, s)
    return count

def main():
    input = sys.stdin.read
    data = input().split()

    n = int(data[0])
    k = int(data[1])
    s = data[2]

    result = replace_and_count(s, k)
    print(result)

if __name__ == "__main__":
    main()