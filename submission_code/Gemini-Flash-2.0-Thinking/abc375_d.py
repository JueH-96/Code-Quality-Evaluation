def solve():
    S = input()
    n = len(S)
    count = 0
    for j in range(1, n - 1):
        left_counts = {}
        for i in range(j):
            char = S[i]
            left_counts[char] = left_counts.get(char, 0) + 1

        right_counts = {}
        for k in range(j + 1, n):
            char = S[k]
            right_counts[char] = right_counts.get(char, 0) + 1

        for char_code in range(ord('A'), ord('Z') + 1):
            char = chr(char_code)
            count += left_counts.get(char, 0) * right_counts.get(char, 0)
    print(count)

# YOUR CODE HERE
solve()