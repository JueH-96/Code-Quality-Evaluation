def max_partitions_after_operations(s: str, k: int) -> int:
    n = len(s)
    if k == 0:
        return 0
    max_result = compute_partitions(s)
    
    for i in range(n):
        for c in range(ord('a'), ord('z') + 1):
            char = chr(c)
            modified = list(s)
            modified[i] = char
            modified_str = ''.join(modified)
            current = compute_partitions(modified_str)
            if current > max_result:
                max_result = current
    return max_result

def compute_partitions(s: str) -> int:
    n = len(s)
    if n == 0:
        return 0
    distinct = set()
    left = 0
    result = 0
    for right in range(n):
        char = s[right]
        if char not in distinct:
            distinct.add(char)
        while len(distinct) > k:
            left_char = s[left]
            if left_char in distinct:
                distinct.remove(left_char)
            left += 1
        result += 1
    return result

def main():
    import sys
    s = sys.stdin.readline().strip()
    k = int(sys.stdin.readline())
    print(max_partitions_after_operations(s, k))

if __name__ == "__main__":
    main()