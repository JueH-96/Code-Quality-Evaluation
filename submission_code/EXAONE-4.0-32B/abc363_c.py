import sys
from collections import Counter

def main():
    data = sys.stdin.read().splitlines()
    n, k = map(int, data[0].split())
    s = data[1].strip()
    freq = Counter(s)
    total_count = 0

    def backtrack(current_list):
        nonlocal total_count
        if len(current_list) == n:
            valid = True
            for i in range(0, n - k + 1):
                left = i
                right = i + k - 1
                is_pal = True
                while left < right:
                    if current_list[left] != current_list[right]:
                        is_pal = False
                        break
                    left += 1
                    right -= 1
                if is_pal:
                    valid = False
                    break
            if valid:
                total_count += 1
            return
        
        for char in list(freq.keys()):
            if freq[char] > 0:
                freq[char] -= 1
                current_list.append(char)
                backtrack(current_list)
                current_list.pop()
                freq[char] += 1
                
    backtrack([])
    print(total_count)

if __name__ == '__main__':
    main()