import sys

def main():
    T = sys.stdin.readline().strip()
    N = int(sys.stdin.readline())
    bags = []
    for _ in range(N):
        parts = sys.stdin.readline().split()
        A_i = int(parts[0])
        strs = parts[1:]
        bags.append(strs)
    
    len_T = len(T)
    INF = float('inf')
    
    # Initialize DP array
    prev = [INF] * (len_T + 1)
    prev[0] = 0  # Starting with empty string
    
    for bag in bags:
        curr = list(prev)  # Start by doing nothing
        for s in bag:
            l = len(s)
            max_pos = len_T - l
            if max_pos < 0:
                continue  # s is longer than remaining T
            for pos in range(0, max_pos + 1):
                if T[pos:pos + l] == s:
                    new_pos = pos + l
                    if prev[pos] + 1 < curr[new_pos]:
                        curr[new_pos] = prev[pos] + 1
        prev = curr
    
    result = prev[len_T]
    print(-1 if result == INF else result)

if __name__ == "__main__":
    main()