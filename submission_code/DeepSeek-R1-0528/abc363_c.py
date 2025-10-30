import itertools
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n, k = map(int, data[0].split())
    s = data[1].strip()
    perms = set(itertools.permutations(s))
    total_valid = 0
    for p in perms:
        found_palindrome = False
        for i in range(0, n - k + 1):
            left = i
            right = i + k - 1
            is_pal = True
            while left < right:
                if p[left] != p[right]:
                    is_pal = False
                    break
                left += 1
                right -= 1
            if is_pal:
                found_palindrome = True
                break
        if not found_palindrome:
            total_valid += 1
    print(total_valid)

if __name__ == "__main__":
    main()