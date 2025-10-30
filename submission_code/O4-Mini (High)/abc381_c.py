import sys

def main():
    input = sys.stdin.readline
    n = int(input().strip())
    s = input().strip()

    # L[i]: number of consecutive '1's ending at i
    L = [0] * n
    for i, ch in enumerate(s):
        if ch == '1':
            L[i] = (L[i-1] if i > 0 else 0) + 1

    # R[i]: number of consecutive '2's starting at i
    R = [0] * n
    for i in range(n-1, -1, -1):
        if s[i] == '2':
            R[i] = (R[i+1] if i+1 < n else 0) + 1

    ans = 1  # minimal answer is 1 because there's at least one '/'
    for i, ch in enumerate(s):
        if ch == '/':
            left = L[i-1] if i > 0 else 0
            right = R[i+1] if i+1 < n else 0
            k = left if left < right else right
            length = 2*k + 1
            if length > ans:
                ans = length

    print(ans)

if __name__ == "__main__":
    main()