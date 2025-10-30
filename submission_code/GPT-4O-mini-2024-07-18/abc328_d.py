def remove_abc(s):
    while 'ABC' in s:
        s = s.replace('ABC', '', 1)
    return s

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    result = remove_abc(S)
    print(result)