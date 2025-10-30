import sys

def main() -> None:
    input = sys.stdin.readline
    N = int(input())
    
    # For every flavor keep the largest and second-largest deliciousness
    top1 = [0] * (N + 1)   # largest
    top2 = [0] * (N + 1)   # second largest
    
    for _ in range(N):
        f, s = map(int, input().split())
        if s > top1[f]:
            top2[f] = top1[f]
            top1[f] = s
        elif s > top2[f]:
            top2[f] = s
    
    # Case 1 : two cups with the same flavor
    max_same = 0
    single_best = []
    for f in range(1, N + 1):
        if top1[f]:
            single_best.append(top1[f])      # best cup of this flavor
        if top2[f]:
            cand = top1[f] + top2[f] // 2    # need at least two cups
            if cand > max_same:
                max_same = cand
    
    # Case 2 : two cups with different flavors
    single_best.sort(reverse=True)
    max_diff = 0
    if len(single_best) >= 2:
        max_diff = single_best[0] + single_best[1]
    
    print(max(max_same, max_diff))

if __name__ == "__main__":
    main()