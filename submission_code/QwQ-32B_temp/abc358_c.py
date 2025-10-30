import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    stands = []
    for _ in range(n):
        s = sys.stdin.readline().strip()
        flavors = set()
        for j in range(m):
            if s[j] == 'o':
                flavors.add(j)
        stands.append(flavors)
    
    min_ans = n  # Initialize with maximum possible value

    # Iterate over all possible non-empty subsets of stands
    for mask in range(1, 1 << n):
        combined = set()
        count = 0
        for i in range(n):
            if (mask >> i) & 1:
                combined |= stands[i]
                count += 1
        if len(combined) == m and count < min_ans:
            min_ans = count
    
    print(min_ans)

if __name__ == "__main__":
    main()