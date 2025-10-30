def can_be_original(s, t):
    if s == t:
        return True
    if len(s) + 1 == len(t):
        for i in range(len(t)):
            if s == t[:i] + t[i+1:]:
                return True
    if len(s) - 1 == len(t):
        for i in range(len(s)):
            if t == s[:i] + s[i+1:]:
                return True
    if len(s) == len(t):
        diff = 0
        for i in range(len(s)):
            if s[i] != t[i]:
                diff += 1
            if diff > 1:
                break
        if diff == 1:
            return True
    return False

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = data[1]
    S = data[2:]
    
    result = []
    for i in range(N):
        if can_be_original(S[i], T):
            result.append(i + 1)
    
    print(len(result))
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()