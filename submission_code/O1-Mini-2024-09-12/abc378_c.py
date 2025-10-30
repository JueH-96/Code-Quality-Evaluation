import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    last_seen = {}
    B = []
    for idx, num in enumerate(A, 1):  # 1-based indexing
        if num in last_seen:
            B.append(str(last_seen[num]))
        else:
            B.append("-1")
        last_seen[num] = idx
    print(' '.join(B))

if __name__ == "__main__":
    main()