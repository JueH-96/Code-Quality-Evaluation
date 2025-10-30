import sys

def is_11_22_string(s):
    n = len(s)
    if n % 2 == 0:
        return False
    mid = (n + 1) // 2 - 1
    if s[mid] != '/':
        return False
    for i in range(mid):
        if s[i] != '1':
            return False
    for i in range(mid + 1, n):
        if s[i] != '2':
            return False
    return True

def max_11_22_substring_length(s):
    n = len(s)
    max_length = 0
    for i in range(n):
        for j in range(i + 1, n + 1):
            substring = s[i:j]
            if is_11_22_string(substring):
                max_length = max(max_length, len(substring))
    return max_length

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    S = data[1]

    result = max_11_22_substring_length(S)
    print(result)

if __name__ == "__main__":
    main()