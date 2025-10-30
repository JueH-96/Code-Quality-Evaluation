def longest_common_prefix(s1, s2):
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return i

def solve(strings):
    n = len(strings)
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += longest_common_prefix(strings[i], strings[j])
    return total

def main():
    n = int(input())
    strings = [input() for _ in range(n)]
    print(solve(strings))

if __name__ == "__main__":
    main()