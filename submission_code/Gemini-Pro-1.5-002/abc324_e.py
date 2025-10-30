# YOUR CODE HERE
def solve():
    n, t = input().split()
    s = [input() for _ in range(int(n))]
    
    def is_subsequence(s, t):
        i = 0
        j = 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                j += 1
            i += 1
        return j == len(t)

    count = 0
    for i in range(int(n)):
        for j in range(int(n)):
            combined = s[i] + s[j]
            if is_subsequence(combined, t):
                count += 1
    print(count)

solve()