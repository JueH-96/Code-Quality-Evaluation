from collections import defaultdict

def check(s, t):
    if len(s) == len(t):
        diff = sum(a != b for a, b in zip(s, t))
        return diff <= 1
    elif len(s) + 1 == len(t):
        for i in range(len(t)):
            if s == t[:i] + t[i + 1:]:
                return True
        return False
    elif len(s) - 1 == len(t):
        for i in range(len(s)):
            if t == s[:i] + s[i + 1:]:
                return True
        return False
    else:
        return False

def main():
    N = int(input())
    T = input()
    S = [input() for _ in range(N)]
    ans = []
    for i, s in enumerate(S):
        if check(s, T):
            ans.append(i + 1)
    print(len(ans))
    print(*ans)

if __name__ == "__main__":
    main()